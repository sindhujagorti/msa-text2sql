# msa-text2sql (spider)

This repo contains baseline Text-to-SQL inference scripts for the SPIDER dataset.

## Data
SPIDER is stored outside the repo (Google Drive or local). See `data/README.md`.

## Baselines (order)
B) cssupport/t5-small-awesome-text-to-sql  
C) suriya7/t5-base-text-to-sql  
A) tscholak/1zha5ono  

## Quick start (Colab or local)
Install:
pip install -r requirements.txt

Run one model:
python baselines/infer_text2sql.py \
  --model cssupport/t5-small-awesome-text-to-sql \
  --spider_dir /path/to/spider \
  --split dev \
  --n 100 \
  --out outputs/preds_B.jsonl
