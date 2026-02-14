BS=32
WR=0.1
ALW=0.1
LR=0.001
TOP_K=5
ATT=8
SD=66
DP=0
WD=0
LS=0

python tuning.py \
    --initial_lr "${LR}" \
    --batch_size "${BS}" \
    --seed "${SD}" \
    --num_heads "${ATT}" \
    --aux_loss_weight "${ALW}" \
    --warmup_ratio "${WR}" \
    --dropout_rate "${DP}" \
    --weight_decay "${WD}" \
    --label_smoothing "${LS}" \
    --top_k ${TOP_K}         

echo "所有实验均已完成。"
echo "你可以使用 'fitlog log logs' 启动可视化服务查看结果。"
