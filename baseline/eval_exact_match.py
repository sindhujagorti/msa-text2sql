import argparse
import json
import re

KEYWORD_CANON = {
    "count ( * )": "count(*)",
    "count( * )": "count(*)",
    "count ( * )": "count(*)",
    "distinct (": "distinct(",
}

def _strip_sql_comments(sql: str) -> str:
    # remove -- ... endline
    sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    # remove /* ... */
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql

def _standardize_quotes(sql: str) -> str:
    # Normalize fancy quotes to normal quotes
    sql = sql.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return sql

def normalize_sql(sql: str) -> str:
    if not sql:
        return ""

    sql = _standardize_quotes(sql)
    sql = _strip_sql_comments(sql)

    # Some HF models echo the prompt. Keep only after the last "sql:" if present.
    # (Safe: if "sql:" not present, nothing changes)
    parts = re.split(r"\bsql\s*:\s*", sql, flags=re.IGNORECASE)
    if len(parts) > 1:
        sql = parts[-1]

    sql = sql.strip().rstrip(";").strip()
    sql = sql.lower()

    # Collapse whitespace
    sql = re.sub(r"\s+", " ", sql)

    # Remove spaces around punctuation/operators that frequently vary
    sql = re.sub(r"\s*,\s*", ",", sql)
    sql = re.sub(r"\s*\(\s*", "(", sql)
    sql = re.sub(r"\s*\)\s*", ")", sql)
    sql = re.sub(r"\s*=\s*", "=", sql)
    sql = re.sub(r"\s*<\s*", "<", sql)
    sql = re.sub(r"\s*>\s*", ">", sql)

    # Canonicalize a few common patterns
    for k, v in KEYWORD_CANON.items():
        sql = sql.replace(k, v)

    # Remove redundant outer whitespace again
    sql = sql.strip()
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
            pred = rec.get("pred_sql", "") or ""
            gold = rec.get("gold_sql", "") or ""
            total += 1
            if exact_match(pred, gold):
                correct += 1

    acc = (correct / total) * 100 if total else 0.0
    print(args.preds)
    print(f"Normalized Exact Match: {correct}/{total} = {acc:.2f}%")

if __name__ == "__main__":
    main()
