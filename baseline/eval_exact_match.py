import argparse
import json
import re


def normalize_sql(sql: str) -> str:
    if sql is None:
        return ""
    sql = sql.strip().lower()
    sql = re.sub(r"\s+", " ", sql)      # collapse whitespace
    sql = sql.rstrip(";")               # remove trailing semicolon
    return sql


def exact_match(pred: str, gold: str) -> bool:
    return normalize_sql(pred) == normalize_sql(gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="Path to preds_*.jsonl")
    args = ap.parse_args()

    total = 0
    correct = 0

    with open(args.preds, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pred = rec.get("pred_sql", "")
            gold = rec.get("gold_sql", "")
            total += 1
            if exact_match(pred, gold):
                correct += 1

    acc = (correct / total) * 100 if total else 0.0
    print(f"{args.preds}")
    print(f"Exact Match: {correct}/{total} = {acc:.2f}%")

if __name__ == "__main__":
    main()
