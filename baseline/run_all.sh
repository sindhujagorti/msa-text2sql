#!/usr/bin/env bash
set -e

SPIDER_DIR=""
N=100
SPLIT="dev"

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --spider_dir)
      SPIDER_DIR="$2"
      shift; shift
      ;;
    --n)
      N="$2"
      shift; shift
      ;;
    --split)
      SPLIT="$2"
      shift; shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$SPIDER_DIR" ]]; then
  echo "Missing --spider_dir"
  exit 1
fi

mkdir -p outputs

echo "Running B..."
python baselines/infer_text2sql.py \
  --model cssupport/t5-small-awesome-text-to-sql \
  --spider_dir "$SPIDER_DIR" \
  --split "$SPLIT" \
  --n "$N" \
  --out outputs/preds_B_t5small.jsonl

echo "Running C..."
python baselines/infer_text2sql.py \
  --model suriya7/t5-base-text-to-sql \
  --spider_dir "$SPIDER_DIR" \
  --split "$SPLIT" \
  --n "$N" \
  --out outputs/preds_C_t5base.jsonl

echo "Running A..."
python baselines/infer_text2sql.py \
  --model tscholak/1zha5ono \
  --spider_dir "$SPIDER_DIR" \
  --split "$SPLIT" \
  --n "$N" \
  --out outputs/preds_A_spider_t5.jsonl

echo "Done. Outputs saved in outputs/"
