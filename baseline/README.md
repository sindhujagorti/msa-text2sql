# Baselines

## Deterministic decoding (reproducibility)
We run deterministic greedy decoding:
- do_sample = false
- num_beams = 1
- temperature = 0.0

## Models (order)
B) cssupport/t5-small-awesome-text-to-sql  
C) suriya7/t5-base-text-to-sql  
A) tscholak/1zha5ono  

## Run all
bash baselines/run_all.sh --spider_dir /path/to/spider --n 100

## Output format
Each line in the JSONL output includes:
- model
- db_id
- question
- gold_sql
- pred_sql
- prompt_style
