#!/usr/bin/env bash
set -e

SPIDER_DIR=""
N=100
SPLIT="dev"
ONLY="all"   # all | B | C | A
DO_EVAL=1

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
    --only)
      ONLY="$2"
      shift; shift
      ;;
    --no_eval)
      DO_EVAL=0
      shift
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

run_model () {
  local label="$1"
  local model="$2"
  local out="$3"

  echo "Running $label..."
  python baseline/infer_text2sql.py \
    --model "$model" \
    --spider_dir "$SPIDER_DIR" \
    --split "$SPLIT" \
    --n "$N" \
    --out "$out"

  if [[ "$DO_EVAL" -eq 1 ]]; then
    echo "Normalized exact match ($label)..."
    python baseline/eval_exact_match.py --preds "$out"

    echo "Execution eval ($label)..."
    python baseline/eval_exec.py \
      --preds "$out" \
      --spider_dir "$SPIDER_DIR" \
      --out_eval "outputs/exec_eval_${label}.jsonl"

    echo "Error analysis ($label)..."
    python baseline/error_analysis.py \
      --exec_eval "outputs/exec_eval_${label}.jsonl" \
      --show 3
  fi
}

if [[ "$ONLY" == "all" || "$ONLY" == "B" ]]; then
  run_model "B_t5small" "cssupport/t5-small-awesome-text-to-sql" "outputs/preds_B_t5small.jsonl"
fi

if [[ "$ONLY" == "all" || "$ONLY" == "C" ]]; then
  run_model "C_t5base" "suriya7/t5-base-text-to-sql" "outputs/preds_C_t5base.jsonl"
fi

if [[ "$ONLY" == "all" || "$ONLY" == "A" ]]; then
  run_model "A_spider_t5" "tscholak/1zha5ono" "outputs/preds_A_spider_t5.jsonl"
fi

echo "Done. Outputs saved in outputs/"
