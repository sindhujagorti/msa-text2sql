import argparse
import json
from collections import Counter, defaultdict
from typing import Dict, Any

def bucket(rec: Dict[str, Any]) -> str:
    status = rec.get("status", "")
    pred_sql = (rec.get("pred_sql") or "").lower()
    err = (rec.get("error") or "").lower()

    if status == "match":
        return "correct"
    if status in {"missing_db", "db_connect_error"}:
        return "db_issue"
    if status == "empty_pred":
        return "empty_pred"
    if status == "gold_error":
        return "gold_sql_error"

    if status == "pred_error":
        if "no such table" in err or "no such column" in err or "ambiguous column" in err:
            return "schema_grounding"
        if "syntax error" in err or "parse" in err:
            return "syntax"
        return "runtime_error"

    # mismatch buckets based on SQL patterns
    nestedish = (pred_sql.count("select") >= 2) or any(k in pred_sql for k in [" in (select", " exists", " union ", " intersect ", " except "])
    aggish = any(k in pred_sql for k in ["count(", "sum(", "avg(", "min(", "max(", "group by", "having"])
    joinish = (" join " in pred_sql) or (pred_sql.count(" from ") >= 1 and pred_sql.count(",") >= 1)
    filterish = " where " in pred_sql

    if nestedish:
        return "nested_or_setops"
    if joinish:
        return "joins"
    if aggish:
        return "aggregation"
    if filterish:
        return "filters"
    return "other_mismatch"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec_eval", required=True)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    counts = Counter()
    examples = defaultdict(list)
    total = 0
    correct = 0

    with open(args.exec_eval, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            b = bucket(rec)
            counts[b] += 1
            total += 1
            if b == "correct":
                correct += 1
            if len(examples[b]) < args.show:
                examples[b].append(rec)

    print(f"File: {args.exec_eval}")
    print(f"Total: {total}")
    print(f"Correct: {correct}/{total} = {(correct/total*100 if total else 0):.2f}%\n")

    print("Bucket counts:")
    for k, v in counts.most_common():
        print(f"  {k:18s} {v}")

    print("\nExamples:")
    for bucket_name, _ in counts.most_common():
        print(f"\n== {bucket_name} ==")
        for r in examples[bucket_name]:
            print("DB:", r.get("db_id"))
            print("Q :", r.get("question"))
            print("G :", r.get("gold_sql"))
            print("P :", r.get("pred_sql"))
            if r.get("error"):
                print("E :", r.get("error"))
            print("-" * 60)

if __name__ == "__main__":
    main()
