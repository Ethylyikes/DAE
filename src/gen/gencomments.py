import os
import json
import openai
import base64
import csv
import time
from PIL import Image
import io

openai.api_base = ""
openai.api_key = ""

           
json_file = '/opt/dataset/pheme_output.json'
image_folder = '/opt/dataset/politifact_images/'
output_csv = '/opt/genpro.csv'
error_csv = '/opt/errorpro.csv'

            
try:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"成功读取JSON文件 {json_file}")
except Exception as e:
    print(f"读取JSON文件时出错: {e}")
    raise

          
roles_list = [
    {"role_number": 1, "gender": "Male",   "age": "Youth (18-35 years old)",    "education": "Bachelor’s degree"},
    {"role_number": 2, "gender": "Male",   "age": "Middle-aged (36-65 years old)", "education": "Postgraduate education"},
    {"role_number": 3, "gender": "Female", "age": "Elderly (over 65 years old)",   "education": "Below Bachelor’s"}
]

                   
processed_ids = set()
if os.path.exists(output_csv):
    try:
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'id' in row:
                    processed_ids.add(row['id'])
    except Exception as e:
        print(f"读取 {output_csv} 时出错: {e}")

                 
file_exists = os.path.exists(output_csv)
try:
    csvfile = open(output_csv, 'a', newline='', encoding='utf-8')
    fieldnames = ['id', 'comments']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
except Exception as e:
    print(f"打开CSV文件 {output_csv} 失败: {e}")
    raise

                 
error_file_exists = os.path.exists(error_csv)
try:
    error_csvfile = open(error_csv, 'a', newline='', encoding='utf-8')
    error_fieldnames = ['id', 'error_message', 'raw_response']
    error_writer = csv.DictWriter(error_csvfile, fieldnames=error_fieldnames)
    if not error_file_exists:
        error_writer.writeheader()
except Exception as e:
    print(f"打开错误记录CSV文件 {error_csv} 失败: {e}")
    raise

processed_count = 0
MAX_RETRIES = 5

def resize_image_to_base64(image_path, max_width=200):
    """
    将图片缩放到指定最大宽度(保持纵横比不变)，然后转换为 Base64 编码并返回。
    如果原图宽度小于等于 max_width，则不进行缩放。
    若发现图像是 RGBA 模式（带透明通道），则先转成 RGB，以便正常写为 JPEG。
    """
    with Image.open(image_path) as img:
              
        if img.width > max_width:
            ratio = max_width / float(img.width)
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

                               
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return encoded_string

def find_image_file(folder, news_id):
    """
    在指定folder中尝试查找和news_id同名且后缀在extensions列表中的任意文件。
    若找到则返回该文件完整路径，否则返回None
    """
    extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    for ext in extensions:
        candidate = os.path.join(folder, news_id + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


                  
for entry in data:
    news_id = entry.get('id')
    if not news_id:
        continue               

                   
    if news_id in processed_ids:
        continue

    title = entry.get('text', '')

              
    comments = entry.get('comments', [])
    comments = comments[:15]          

                   
    img_path = find_image_file(image_folder, news_id)
    image_line = ""
    if img_path:
        try:
            resized_base64 = resize_image_to_base64(img_path, max_width=200)
            image_line = f"News Picture: data:image/jpeg;base64,{resized_base64}\n\n"
        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {e}")
                           
            image_line = ""

                  
    comments_text = "\n".join(comments)

                  
    prompt = (
        "You are a training data creator tasked with generating comments for a news article "
        "based on different user profiles. The news article may or may not be reliable, "
        "so each user's comment should reflect their own cognitive and emotional empathy "
        "towards the news. In other words:\n"
        "1. If the user can cognitively and emotionally empathize with the news story, "
        "   their comment should show support or agreement.\n"
        "2. If the user cannot align with the news story—whether due to distrust, "
        "   skepticism, or lack of emotional resonance—their comment should express "
        "   doubt, disagreement, or concern.\n\n"
        "Cognitive empathy: The comment demonstrates understanding of the news content, possibly with skepticism or disagreement, "
        "but still recognizes the core issue."
        "Emotional empathy: The comment expresses compassion, support, or emotional resonance with the news."
        f"News Title: {title}\n"
        f"{image_line}"
        f"Existing Comments:\n{comments_text}\n\n"
        "User Profiles:\n"
    )

              
    for role in roles_list:
        prompt += (
            f"Profile {role['role_number']}: "
            f"Gender: {role['gender']}, "
            f"Age: {role['age']}, "
            f"Education: {role['education']}\n"
        )

    prompt += (
        "\nPlease generate one comment for each profile listed above. "
        "Each comment should:\n"
        "- Reflect the user's demographic background.\n"
        "- Show how the user either resonates with or rejects the news, "
        "  based on their cognitive and emotional empathy.\n\n"
        "Important Formatting Instructions:\n"
        "1. Do NOT include any numbering, bullet points, or labels (e.g., 'Profile 1:', '1.', etc.) "
        "   before each comment.\n"
        "2. Output each comment on a new line.\n"
        "3. Do not include any additional text or formatting other than the comments themselves.\n\n"
        "Now, generate the comments accordingly:"
    )

            
    retries = 0
    generated_comments = ""

    while retries < MAX_RETRIES:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",            
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates comments based on user profiles."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=4096,
                request_timeout=360
            )

                            
            try:
                generated_comments = response.choices[0].message.content.strip()
            except (KeyError, AttributeError) as err:
                                               
                error_writer.writerow({
                    'id': news_id,
                    'error_message': f"{type(err).__name__}: {str(err)}",
                    'raw_response': str(response)
                })
                error_csvfile.flush()
                break                 

                                    
            break

        except openai.error.RateLimitError as e:
                              
            wait_time = 2 ** retries
            print(f"Rate limit reached. 等待 {wait_time} 秒后重试。错误信息: {e}")
            time.sleep(wait_time)
            retries += 1

        except openai.error.OpenAIError as e:
            print(f"调用GPT API生成新闻ID {news_id} 的评论时出错。错误信息: {e}")
                                  
            error_writer.writerow({
                'id': news_id,
                'error_message': f"OpenAIError: {str(e)}",
                'raw_response': ""
            })
            error_csvfile.flush()
                         
            break

                                      
    if not generated_comments:
        print(f"未能为新闻ID {news_id} 生成评论（已记录错误或被跳过）。")
        continue

                     
    try:
        writer.writerow({
            'id': news_id,
            'comments': generated_comments
        })
        csvfile.flush()
    except Exception as e:
        print(f"写入CSV失败: {e}")

    processed_ids.add(news_id)
    processed_count += 1
    print(f"已处理新闻ID: {news_id} | 已处理数量: {processed_count}")

                      
    time.sleep(1)

      
csvfile.close()
error_csvfile.close()

print("操作完成。")
