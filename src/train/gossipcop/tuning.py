import os
import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel, AutoImageProcessor, SwinModel
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import numpy as np
import json
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from torch.cuda.amp import autocast, GradScaler
import random
import logging
import argparse

                                   
import fitlog

                                                                                
             
                                                                                
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)                        
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

                                                                                
                         
                                                                                
parser = argparse.ArgumentParser()
parser.add_argument('--data_json', type=str, default='/opt/dataset/gossipcop_gpt_cleaned.json',
                    help='数据文件json路径')
parser.add_argument('--image_folder', type=str, default='/opt/dataset/gossipcop_image/',
                    help='图片文件夹路径')
parser.add_argument('--vis_dir', type=str, default='/opt/loss_visualization_image/',
                    help='保存训练日志或可视化csv的目录')
parser.add_argument('--model_dir', type=str, default='/opt/modeled/',
                    help='保存模型检查点的目录')
parser.add_argument('--roberta_model_dir', type=str, default='/opt/download_model/',
                    help='Roberta模型所在路径')
parser.add_argument('--swin_model_dir', type=str, default='/opt/swin/',
                    help='Swin模型所在路径')
parser.add_argument('--gpt_csv_file', type=str, default='/opt/gengp.csv',
                    help='GPT生成评论的csv文件路径')

parser.add_argument('--initial_lr', type=float, default=0.001,
                    help='初始学习率')
parser.add_argument('--batch_size', type=int, default=32,
                    help='批处理大小')
parser.add_argument('--top_k', type=int, default=3,
                    help='Pointer Network中选取评论的top_k')
parser.add_argument('--aux_loss_weight', type=float, default=0.1,
                    help='辅助损失权重')
parser.add_argument('--max_epochs', type=int, default=1000,
                    help='最大训练轮数')
parser.add_argument('--warmup_ratio', type=float, default=0.1,
                    help='学习率预热比例')
parser.add_argument('--seed', type=int, default=3407,
                    help='随机种子')
parser.add_argument('--num_heads', type=int, default=8,
                    help='多头注意力头数')

       
parser.add_argument('--dropout_rate', type=float, default=0.1,
                    help='Dropout概率')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='L2正则化权重衰减系数')
parser.add_argument('--label_smoothing', type=float, default=0.0,
                    help='标签平滑系数')

args = parser.parse_args()

                                             
fitlog.set_log_dir("logs/")
                  
fitlog.add_hyper(args)
                      
fitlog.add_hyper_in_file(__file__)

                                                                                
                  
                                                                                
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

                                                                                
                            
                                                                                
data_json = args.data_json
image_folder = args.image_folder
vis_dir = args.vis_dir
model_dir = args.model_dir
roberta_model_dir = args.roberta_model_dir
swin_model_dir = args.swin_model_dir
gpt_csv_file = args.gpt_csv_file

initial_lr = args.initial_lr
batch_size = args.batch_size
top_k = args.top_k
aux_loss_weight = args.aux_loss_weight
max_epochs = args.max_epochs
num_heads = args.num_heads           

dropout_rate = args.dropout_rate
weight_decay = args.weight_decay
label_smoothing = args.label_smoothing

max_patience = 5          
best_accuracy = 0.0
best_loss = float('inf')
no_acc_improve = 0
no_loss_improve = 0

best_model_path = os.path.join(model_dir, 'best_model.pth')

                                                                                
        
                                                                                
class CustomDataset(Dataset):
    def __init__(self, json_file, image_folder, gen_csv_file=None):
        self.image_folder = image_folder
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.gpt_comments_dict = {}
        if gen_csv_file is not None and os.path.exists(gen_csv_file):
            df_gpt = pd.read_csv(gen_csv_file)
            for idx, row in df_gpt.iterrows():
                data_id = str(row['id'])
                raw_comments_str = str(row['comments'])
                splitted_comments = raw_comments_str.split('\n\n')
                splitted_comments = [c.strip() for c in splitted_comments if c.strip()]
                self.gpt_comments_dict[data_id] = splitted_comments

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        data_id = str(item['id'])
        text = item['text']
        label = item['label']

                
        comments = item['comments']
        if isinstance(comments, str):
                                 
            comments = [c.strip() for c in comments.split("\n") if c.strip()]
        elif isinstance(comments, list):
            comments = [str(c).strip() if c is not None else "" for c in comments]
        else:
            comments = []

                  
        comments = comments[:15]
                         
        comments = [c.replace('@user', '').strip() for c in comments]

                              
        gpt_comments = self.gpt_comments_dict.get(data_id, [])
        gpt_comments = [c.replace('@user', '').strip() for c in gpt_comments]

                                                
        merged_comments = comments + gpt_comments
        merged_comments = merged_comments[:15]
                

              
        image_path = os.path.join(self.image_folder, f"{data_id}.jpg")
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))

        return data_id, text, merged_comments, image, label


def collate_fn(batch):
    data_ids = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    merged_comments = [item[2] for item in batch]             
    images = [item[3] for item in batch]
    labels = [item[4] for item in batch]

                 
    text_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=512
    )

                                 
    max_length_comment = 128
    max_comments = max(len(comments) for comments in merged_comments)
    batch_input_ids = []
    batch_attention_mask = []
    for comments in merged_comments:
        sample_input_ids = []
        sample_attention_mask = []
        for comment in comments:
            tokens = tokenizer(
                comment,
                padding='max_length',
                truncation=True,
                max_length=max_length_comment,
                return_tensors='pt'
            )
            sample_input_ids.append(tokens['input_ids'].squeeze(0))                        
            sample_attention_mask.append(tokens['attention_mask'].squeeze(0))
                                   
        if len(comments) < max_comments:
            pad_num = max_comments - len(comments)
            for _ in range(pad_num):
                sample_input_ids.append(torch.zeros(max_length_comment, dtype=torch.long))
                sample_attention_mask.append(torch.zeros(max_length_comment, dtype=torch.long))
        sample_input_ids = torch.stack(sample_input_ids, dim=0)                                             
        sample_attention_mask = torch.stack(sample_attention_mask, dim=0)                                       
        batch_input_ids.append(sample_input_ids.unsqueeze(0))
        batch_attention_mask.append(sample_attention_mask.unsqueeze(0))
    batch_input_ids = torch.cat(batch_input_ids, dim=0)                                                
    batch_attention_mask = torch.cat(batch_attention_mask, dim=0)                                        
    comments_inputs = {"input_ids": batch_input_ids, "attention_mask": batch_attention_mask}

            
    images_processed = image_processor(images=images, return_tensors='pt')
    labels = torch.tensor(labels, dtype=torch.long)

    return data_ids, text_inputs, comments_inputs, images_processed, labels


def count_swin_layers(model, target_class_name='SwinLayer'):
    count = 0
    for module in model.modules():
        if module.__class__.__name__ == target_class_name:
            count += 1
    return count


def freeze_swin_layers(model):
    transformer_layer_count = count_swin_layers(model, 'SwinLayer')
    if transformer_layer_count == 0:
        logger.warning("未检测到任何 'SwinLayer' 模块。请检查模型的实际模块名称和类型。")
        for name, module in model.named_modules():
            logger.info(f"{name}: {module.__class__.__name__}")
        return

    swin_layers = [module for module in model.modules() if module.__class__.__name__ == 'SwinLayer']
    layers_to_freeze = swin_layers[:-1]

    for layer in layers_to_freeze:
        for param in layer.parameters():
            param.requires_grad = False


class PointerNetwork(nn.Module):
    def __init__(self, input_dim, k):
        super(PointerNetwork, self).__init__()
        self.k = k
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
                                        
        B, N, input_dim = x.size()
        attn_scores = self.attn(x).squeeze(-1)          
        attn_probs = F.softmax(attn_scores, dim=1)          
        k = min(self.k, N)
        if k > 0:
            topk_probs, topk_indices = torch.topk(attn_probs, k, dim=1)
                                       
            selected = x.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, input_dim))
        else:
            selected = x.new_zeros((B, self.k, input_dim))
            topk_probs = attn_probs.new_zeros((B, self.k))
            topk_indices = attn_probs.new_zeros((B, self.k), dtype=torch.long)
        return selected, attn_probs, topk_indices


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=num_heads)

    def forward(self, x):
                                             
        x = x.transpose(0, 1)                                       
        attn_output, attn_weights = self.attention(x, x, x)
        attn_output = attn_output.transpose(0, 1)                                       
        return attn_output, attn_weights


class CrossAttention(nn.Module):
    def __init__(self, embed_dim_x, embed_dim_y, num_heads=8):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim_x, num_heads=num_heads)

    def forward(self, x, y):
                                                
        x = x.transpose(0, 1)
        y = y.transpose(0, 1)
        attn_output, _ = self.attention(x, y, y)
        attn_output = attn_output.transpose(0, 1)
        return attn_output


class MultimodalModel(nn.Module):
    def __init__(self, roberta_model_text, roberta_model_comments, swin_model,
                 top_k=5, num_classes=2, num_heads=8, dropout_rate=0.1):
        super(MultimodalModel, self).__init__()
        self.roberta_model_text = roberta_model_text
        self.roberta_model_comments = roberta_model_comments
        self.swin_model = swin_model
        self.top_k = top_k
        self.num_classes = num_classes

        roberta_hidden_size = roberta_model_text.config.hidden_size
        swin_output_size = swin_model.config.hidden_size

        self.text_proj = nn.Linear(roberta_hidden_size, 512)
        self.comments_proj = nn.Linear(roberta_hidden_size, 512)
        self.image_proj = nn.Linear(swin_output_size, 512)
                  
        self.dropout = nn.Dropout(dropout_rate)

                               
        self.self_attention_text = SelfAttention(512, num_heads=num_heads)
        self.self_attention_image = SelfAttention(512, num_heads=num_heads)

        self.self_attention_comments = SelfAttention(512)
        self.pointer_network = PointerNetwork(512, top_k)
        self.self_attention_comments = SelfAttention(512, num_heads=num_heads)

                                        
        self.cross_attention = CrossAttention(512, 512, num_heads=num_heads)

        self.max_pooling = nn.AdaptiveMaxPool1d(1)
        self.mean_pooling = nn.AdaptiveAvgPool1d(1)

        self.fc_concat = nn.Linear(3584, 512)
        self.fc_final = nn.Linear(512, num_classes)

    def forward(self, text_input, comments_input, image_input):
              
        text_output = self.roberta_model_text(**text_input)
        text_features = text_output.last_hidden_state                             
        text_features = self.text_proj(text_features)                      
        text_features = self.dropout(text_features)
        text_features, _ = self.self_attention_text(text_features)

              
                                                    
        B, N, L = comments_input["input_ids"].shape
        flattened_input_ids = comments_input["input_ids"].view(B * N, L)
        flattened_attention_mask = comments_input["attention_mask"].view(B * N, L)
                            
        comments_output = self.roberta_model_comments(input_ids=flattened_input_ids,
                                                      attention_mask=flattened_attention_mask)
                                      
        cls_tokens = comments_output.last_hidden_state[:, 0, :]                      
        cls_tokens = self.comments_proj(cls_tokens)              
                                               
                         
        comments_features = cls_tokens.view(B, N, -1)
                  
        comments_features, _ = self.self_attention_comments(comments_features)
                                         
        selected_comments, attn_probs, topk_indices = self.pointer_network(comments_features)
                           
        selected_comments, comments_attn_weights = self.self_attention_comments(selected_comments)

              
        image_output = self.swin_model(**image_input)
        image_features = image_output.last_hidden_state                             
        image_features = self.image_proj(image_features)                      
        image_features = self.dropout(image_features)
        image_features, _ = self.self_attention_image(image_features)

                                   
        cross_attention_features = self.cross_attention(text_features, image_features)
        pooled_text = self.mean_pooling(text_features.transpose(1, 2)).squeeze(-1)
        pooled_comments = self.max_pooling(selected_comments.transpose(1, 2)).squeeze(-1)
        pooled_cross_attention = self.mean_pooling(cross_attention_features.transpose(1, 2)).squeeze(-1)

        a1 = torch.cat((pooled_text, pooled_comments), dim=-1)
        a3 = torch.cat((a1, pooled_cross_attention), dim=-1)
        a4 = torch.abs(pooled_cross_attention - pooled_comments)
        a5 = torch.cat((pooled_cross_attention, a1, a4), dim=-1)
        final_concat = torch.cat((a3, a5), dim=-1)

        combined_features = self.fc_concat(final_concat)
        combined_features = self.dropout(combined_features)
        output = self.fc_final(combined_features)

        return output, comments_attn_weights, attn_probs, topk_indices


                                                                                
                      
                                                                                
logger.info("Loading tokenizer and models...")
tokenizer = RobertaTokenizer.from_pretrained(roberta_model_dir)
roberta_model_text = RobertaModel.from_pretrained(roberta_model_dir)
roberta_model_comments = RobertaModel.from_pretrained(roberta_model_dir)

image_processor = AutoImageProcessor.from_pretrained(swin_model_dir)
swin_model = SwinModel.from_pretrained(swin_model_dir)

freeze_swin_layers(swin_model)

logger.info("Building dataset and dataloaders...")
dataset = CustomDataset(
    json_file=data_json,
    image_folder=image_folder,
    gen_csv_file=gpt_csv_file
)

total_data_size = len(dataset)
indices = list(range(total_data_size))

train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
train_indices, val_indices = train_test_split(train_indices, test_size=0.25, random_state=42)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

                                                                                
                                    
                                                                                
logger.info("Initializing MultimodalModel...")
model = MultimodalModel(
    roberta_model_text,
    roberta_model_comments,
    swin_model,
    top_k=top_k,
    num_classes=2,
    num_heads=num_heads,
    dropout_rate=dropout_rate
)

                        
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

                                                                                
            
                                                                                
                                   
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)

total_steps_per_epoch = len(train_loader)
total_steps = max_epochs * total_steps_per_epoch
warmup_steps = int(args.warmup_ratio * total_steps)                    

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

                 
scaler = GradScaler()

                                                                                
        
                                                                                
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (data_ids, text_inputs, comments_inputs, images_processed, labels) in enumerate(data_loader):
            labels = labels.to(device)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            comments_inputs = {k: v.to(device) for k, v in comments_inputs.items()}
            images_processed = {k: v.to(device) for k, v in images_processed.items()}

            outputs, comments_attn_weights, attn_probs, topk_indices = model(
                text_inputs,
                comments_inputs,
                images_processed
            )

            loss_main = criterion(outputs, labels)
            topk_probs = attn_probs.gather(1, topk_indices)
            loss_aux = 1 - torch.mean(torch.sum(topk_probs, dim=1))
            loss_val = loss_main + aux_loss_weight * loss_aux

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            running_loss += loss_main.item() * labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    eval_loss = running_loss / total
    eval_accuracy = correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

    return eval_loss, eval_accuracy, precision, recall, f1

                                                                                
      
                                                                                
logger.info("Starting training...")
epoch = 0
columns_batch = ['Epoch', 'Batch', 'Mode', 'Loss', 'Main Loss', 'Aux Loss', 'Accuracy']
df_epochs = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Train Accuracy',
                                  'Val Loss', 'Val Accuracy', 'Precision', 'Recall', 'F1-score'])
lr_history = []

while True:
    epoch += 1
    model.train()
    df_batches = pd.DataFrame(columns=columns_batch)
    train_loss_epoch_accumulated = 0.0
    train_correct_epoch_accumulated = 0
    train_total_epoch_accumulated = 0

    for i, (data_ids, text_inputs, comments_inputs, images_processed, labels) in enumerate(
        tqdm(train_loader, desc=f'第 {epoch} 轮训练', leave=False)):

        labels = labels.to(device)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        comments_inputs = {k: v.to(device) for k, v in comments_inputs.items()}
        images_processed = {k: v.to(device) for k, v in images_processed.items()}

              
        with autocast():
            outputs, comments_attn_weights, attn_probs, topk_indices = model(
                text_inputs,
                comments_inputs,
                images_processed
            )

            loss_main = criterion(outputs, labels)
            topk_probs = attn_probs.gather(1, topk_indices)
            loss_aux = 1 - torch.mean(torch.sum(topk_probs, dim=1))
            loss = loss_main + aux_loss_weight * loss_aux

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        current_step = (epoch - 1) * total_steps_per_epoch + i + 1
        lr_history.append({'Step': current_step, 'Learning Rate': scheduler.get_last_lr()[0]})

        _, predicted = torch.max(outputs, 1)
        correct_train = (predicted == labels).sum().item()
        total_train = labels.size(0)

        train_loss_epoch_accumulated += loss_main.item() * total_train
        train_correct_epoch_accumulated += correct_train
        train_total_epoch_accumulated += total_train

        batch_data = {
            'Epoch': epoch,
            'Batch': i + 1,
            'Mode': '训练',
            'Loss': loss.item(),
            'Main Loss': loss_main.item(),
            'Aux Loss': loss_aux.item(),
            'Accuracy': correct_train / total_train
        }
        df_batches = pd.concat([df_batches, pd.DataFrame([batch_data])], ignore_index=True)

                                                          
        fitlog.add_loss(loss.item(), name="train_loss", step=current_step)

        logger.info(f'Epoch {epoch}, Batch {i + 1}, '
                    f'Loss: {loss.item():.4f}, '
                    f'Main Loss: {loss_main.item():.4f}, '
                    f'Aux Loss: {loss_aux.item():.4f}, '
                    f'Train Acc(Batch): {correct_train / total_train * 100:.2f}%, '
                    f'LR: {scheduler.get_last_lr()[0]:.6f}')

    train_loss_epoch_average = train_loss_epoch_accumulated / train_total_epoch_accumulated
    train_accuracy_epoch_average = train_correct_epoch_accumulated / train_total_epoch_accumulated

    val_loss_epoch, val_accuracy_epoch, precision_epoch, recall_epoch, f1_epoch = evaluate_model(model, val_loader)

    epochs_data = {
        'Epoch': epoch,
        'Train Loss': train_loss_epoch_average,
        'Train Accuracy': train_accuracy_epoch_average,
        'Val Loss': val_loss_epoch,
        'Val Accuracy': val_accuracy_epoch,
        'Precision': precision_epoch,
        'Recall': recall_epoch,
        'F1-score': f1_epoch
    }
    df_epochs = pd.concat([df_epochs, pd.DataFrame([epochs_data])], ignore_index=True)

                                             
    fitlog.add_metric(
        {
            "val": {
                "loss": val_loss_epoch,
                "accuracy": val_accuracy_epoch,
                "precision": precision_epoch,
                "recall": recall_epoch,
                "f1": f1_epoch
            }
        },
        step=epoch
    )

    logger.info(f'==> Epoch {epoch} | '
                f'Train Loss: {train_loss_epoch_average:.4f}, Train Accuracy: {train_accuracy_epoch_average:.4f} | '
                f'Val Loss: {val_loss_epoch:.4f}, Val Accuracy: {val_accuracy_epoch:.4f}, '
                f'Precision: {precision_epoch:.4f}, Recall: {recall_epoch:.4f}, F1-score: {f1_epoch:.4f}')

                 
    os.makedirs(vis_dir, exist_ok=True)
    batch_csv_path = os.path.join(vis_dir, f'batch_data_{epoch}.csv')
    df_batches.to_csv(batch_csv_path, index=False)

            
    improved = False
    if val_accuracy_epoch > best_accuracy:
        best_accuracy = val_accuracy_epoch
        no_acc_improve = 0
        improved = True
                                              
        fitlog.add_best_metric({"val": {"best_accuracy": best_accuracy}})
    else:
        no_acc_improve += 1

    if val_loss_epoch < best_loss:
        best_loss = val_loss_epoch
        no_loss_improve = 0
        improved = True
                                                      
        fitlog.add_best_metric({"val": {"best_loss": best_loss}})
    else:
        no_loss_improve += 1

    if improved:
        torch.save(model.state_dict(), best_model_path)

    if no_acc_improve >= max_patience or no_loss_improve >= max_patience:
        logger.info("验证集指标多次未改善，提前停止训练。")
        break

         
lr_csv_path = os.path.join(vis_dir, 'learning_rate_history.csv')
df_lr = pd.DataFrame(lr_history)
df_lr.to_csv(lr_csv_path, index=False)

       
logger.info("Evaluating on Test set with best model...")
model.load_state_dict(torch.load(best_model_path))
final_test_loss, final_test_accuracy, final_precision, final_recall, final_f1 = evaluate_model(model, test_loader)

final_data = {
    'Epoch': 'Final',
    'Train Loss': None,
    'Train Accuracy': None,
    'Val Loss': None,
    'Val Accuracy': None,
    'Precision': final_precision,
    'Recall': final_recall,
    'F1-score': final_f1
}
df_epochs = pd.concat([df_epochs, pd.DataFrame([final_data])], ignore_index=True)

epoch_csv_path = os.path.join(vis_dir, 'epoch_data.csv')
df_epochs.to_csv(epoch_csv_path, index=False)

final_model_path = os.path.join(model_dir, 'model_finished.pth')
torch.save(model.state_dict(), final_model_path)

logger.info("训练完成")
logger.info(f'最终测试集结果 - 测试损失: {final_test_loss:.4f}, 测试准确率: {final_test_accuracy:.4f}, '
            f'Precision: {final_precision:.4f}, Recall: {final_recall:.4f}, F1-score: {final_f1:.4f}')

                                                    
fitlog.add_best_metric({
    "test": {
        "loss": final_test_loss,
        "accuracy": final_test_accuracy,
        "precision": final_precision,
        "recall": final_recall,
        "f1": final_f1
    }
})

def save_test_results(model, data_loader, correct_csv_path, incorrect_csv_path):
    model.eval()
    correct_ids = []
    incorrect_ids = []
    with torch.no_grad():
        for data_ids, text_inputs, comments_inputs, images_processed, labels in data_loader:
            labels = labels.to(device)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            comments_inputs = {k: v.to(device) for k, v in comments_inputs.items()}
            images_processed = {k: v.to(device) for k, v in images_processed.items()}

            outputs, comments_attn_weights, attn_probs, topk_indices = model(
                text_inputs,
                comments_inputs,
                images_processed
            )
            _, predicted = torch.max(outputs.data, 1)
            for did, pred, label in zip(data_ids, predicted.cpu().numpy(), labels.cpu().numpy()):
                if pred == label:
                    correct_ids.append(did)
                else:
                    incorrect_ids.append(did)
    pd.DataFrame({'id': correct_ids}).to_csv(correct_csv_path, index=False)
    pd.DataFrame({'id': incorrect_ids}).to_csv(incorrect_csv_path, index=False)
    logger.info(f"测试集预测正确的样本 id 已保存至 {correct_csv_path}")
    logger.info(f"测试集预测错误的样本 id 已保存至 {incorrect_csv_path}")

correct_csv_path = os.path.join(vis_dir, f'test_correct_ids_ablation.csv')
incorrect_csv_path = os.path.join(vis_dir, f'test_incorrect_ids_ablation.csv')
save_test_results(model, test_loader, correct_csv_path, incorrect_csv_path)

           
fitlog.finish()

