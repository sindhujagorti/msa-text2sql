# eval_exec.py
import argparse
import json
import os
import sqlite3
from typing import Any, List, Tuple

def _connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    # Make results consistent
    con.row_factory = sqlite3.Row
    return con

def _run_query(con: sqlite3.Connection, sql: str) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    cur = con.cursor()
    cur.execute(sql)
    rows = cur.fetchall()

    # Column names (stable across sqlite Row)
    colnames = [d[0] for d in cur.description] if cur.description else []

    # Convert rows to tuples, keep only primitive values
    out_rows = []
    for r in rows:
        if isinstance(r, sqlite3.Row):
            out_rows.append(tuple(r[c] for c in colnames))
        else:
            out_rows.append(tuple(r))
    return colnames, out_rows

def _normalize_value(v: Any) -> Any:
    # Normalize floats a bit (sqlite can produce small float diffs)
    if isinstance(v, float):
        return round(v, 6)
    return v

def results_equivalent(
    gold_cols: List[str],
    gold_rows: List[Tuple[Any, ...]],
    pred_cols: List[str],
    pred_rows: List[Tuple[Any, ...]],
) -> bool:
    # If column counts differ, not equivalent
    if len(gold_cols) != len(pred_cols):
        return False

    # Order-insensitive comparison of rows
    gold_norm = sorted(tuple(_normalize_value(x) for x in row) for row in gold_rows)
    pred_norm = sorted(tuple(_normalize_value(x) for x in row) for row in pred_rows)
    return gold_norm == pred_norm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="Path to preds_*.jsonl")
    ap.add_argument("--spider_dir", required=True, help="Path to local SPIDER directory")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only evaluate first N")
    args = ap.parse_args()

    total = 0
    exec_correct = 0
    pred_error = 0
    gold_error = 0
    missing_db = 0

    with open(args.preds, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            db_id = rec.get("db_id")
            pred_sql = (rec.get("pred_sql") or "").strip()
            gold_sql = (rec.get("gold_sql") or "").strip()

            if not db_id:
                continue

            db_path = os.path.join(args.spider_dir, "database", db_id, f"{db_id}.sqlite")
            if not os.path.exists(db_path):
                missing_db += 1
                continue

            total += 1
            if args.limit and total > args.limit:
                break

            # Basic guard: empty predicted SQL cannot be correct
            if not pred_sql or not gold_sql:
                continue

            try:
                con = _connect(db_path)
            except Exception:
                missing_db += 1
                continue

            try:
                try:
                    gold_cols, gold_rows = _run_query(con, gold_sql)
                except Exception:
                    gold_error += 1
                    continue

                try:
                    pred_cols, pred_rows = _run_query(con, pred_sql)
                except Exception:
                    pred_error += 1
                    continue

                if results_equivalent(gold_cols, gold_rows, pred_cols, pred_rows):
                    exec_correct += 1
            finally:
                con.close()

    acc = (exec_correct / total) * 100 if total else 0.0
    print(args.preds)
    print(f"Execution Accuracy: {exec_correct}/{total} = {acc:.2f}%")
    print(f"Pred SQL errors: {pred_error}")
    print(f"Gold SQL errors: {gold_error}")
    print(f"Missing DB files: {missing_db}")

if __name__ == "__main__":
    main()
