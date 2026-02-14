# DAE
[Information Fusion] Official Implementation of DAE (Bridging Cognition and Emotion: Empathy-Driven Multimodal Misinformation Detection)

# Comment Generation, Cleaning, and Training Guide

This repo contains two stages before training: (1) generate comments with OpenAI, (2) clean / filter the generated comments, and finally (3) train three datasets (GossipCop, PHEME, PolitiFact) with the provided `run.sh` scripts.

## 1) Configure API endpoint and data paths
Update the hard-coded paths before running scripts:
- OpenAI endpoint/keys in `gen/gencomments.py:10-11` and `gen/filtering.py:10-11`.
- Generation inputs/outputs in `gen/gencomments.py:14-17`:
  - `json_file` – input news + image metadata JSON
  - `image_folder` – directory of images
  - `output_csv` – CSV to write generated comments (`id,comments`)
  - `error_csv` – CSV to log failures
- Cleaning inputs/outputs in `gen/filtering.py:17-21`:
  - `json_file`, `image_folder`, `output_csv`, `error_csv` (same meaning as above)
  - `misclassified_csv` – CSV listing `news_id` rows to be re-generated/filtered

## 2) Generate comments
```bash
cd repo/src
python gen/gencomments.py
```

## 3) Clean / filter generated comments
```bash
cd repo/src
python gen/filtering.py
```

## 4) Train models
Each dataset has its own launcher; run with Bash or PowerShell:
- GossipCop: `cd repo/src/train/gossipcop && ./run.sh`
- PHEME: `cd repo/src/train/pheme && ./run.sh`
- PolitiFact: `cd repo/src/train/political && ./run.sh`

### Passing custom data/image/comment paths to training
`run.sh` currently relies on the defaults in each `tuning.py`. If your files are not under `/opt/...`, add overrides to the `python tuning.py` call, for example:
```bash
python tuning.py \
  --data_json "/abs/path/to/gossipcop_clean.json" \
  --image_folder "/abs/path/to/gossipcop_images/" \
  --gpt_csv_file "/abs/path/to/gossipcop_comments.csv" \
  --vis_dir "./logs" \
  --model_dir "./checkpoints" \
  --roberta_model_dir "/abs/path/to/roberta" \
  --swin_model_dir "/abs/path/to/swin" \
  --initial_lr "${LR}" --batch_size "${BS}" --seed "${SD}" \
  --num_heads "${ATT}" --aux_loss_weight "${ALW}" \
  --warmup_ratio "${WR}" --dropout_rate "${DP}" \
  --weight_decay "${WD}" --label_smoothing "${LS}" \
  --top_k ${TOP_K}
```
Apply the same pattern inside `train/pheme/run.sh` and `train/political/run.sh`.

## Tips
- Keep `output_csv` from generation and cleaning aligned with the `--gpt_csv_file` you pass to training.

## 📝 Citation
If you find DAE useful, please star and cite it:
```bibtex
@article{yuan2026bridging,
title = {Bridging cognition and emotion: Empathy-driven multimodal misinformation detection},
journal = {Information Fusion},
volume = {132},
pages = {104210},
year = {2026},
issn = {1566-2535},
doi = {https://doi.org/10.1016/j.inffus.2026.104210},
url = {https://www.sciencedirect.com/science/article/pii/S1566253526000898},
author = {Lu Yuan and Zihan Wang and Zhengxuan Zhang and Lei Shi}
}
```
