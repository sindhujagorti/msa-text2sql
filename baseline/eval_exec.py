import argparse
import json
import os
import sqlite3
from typing import Any, List, Tuple, Optional

def _connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con

def _run_query(con: sqlite3.Connection, sql: str) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    cur = con.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    colnames = [d[0] for d in cur.description] if cur.description else []

    out_rows = []
    for r in rows:
        if isinstance(r, sqlite3.Row):
            out_rows.append(tuple(r[c] for c in colnames))
        else:
            out_rows.append(tuple(r))
    return colnames, out_rows

def _normalize_value(v: Any) -> Any:
    if isinstance(v, float):
        v = round(v, 6)
        if float(v).is_integer():
            return int(v)
        return v
    if isinstance(v, str):
        return v.strip()
    return v

def _has_order_by(sql: str) -> bool:
    return "order by" in (sql or "").lower()

def results_equivalent(
    gold_cols: List[str],
    gold_rows: List[Tuple[Any, ...]],
    pred_cols: List[str],
    pred_rows: List[Tuple[Any, ...]],
    gold_sql: str,
    pred_sql: str,
) -> bool:
    if len(gold_cols) != len(pred_cols):
        return False

    gold_norm = [tuple(_normalize_value(x) for x in row) for row in gold_rows]
    pred_norm = [tuple(_normalize_value(x) for x in row) for row in pred_rows]

    # Respect ordering if either query uses ORDER BY
    if _has_order_by(gold_sql) or _has_order_by(pred_sql):
        return gold_norm == pred_norm

    # Otherwise order-insensitive
    return sorted(gold_norm) == sorted(pred_norm)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="Path to preds_*.jsonl")
    ap.add_argument("--spider_dir", required=True, help="Path to local SPIDER directory")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only evaluate first N")
    ap.add_argument("--out_eval", default="", help="Optional: write per-example exec eval JSONL")
    args = ap.parse_args()

    out_f = None
    if args.out_eval:
        os.makedirs(os.path.dirname(args.out_eval) or ".", exist_ok=True)
        out_f = open(args.out_eval, "w", encoding="utf-8")

    total = 0
    exec_correct = 0
    pred_error = 0
    gold_error = 0
    missing_db = 0
    empty_pred = 0

    with open(args.preds, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            db_id = rec.get("db_id")
            pred_sql = (rec.get("pred_sql") or "").strip()
            gold_sql = (rec.get("gold_sql") or "").strip()
            question = rec.get("question", "")

            if not db_id:
                continue

            db_path = os.path.join(args.spider_dir, "database", db_id, f"{db_id}.sqlite")
            if not os.path.exists(db_path):
                missing_db += 1
                if out_f:
                    out_f.write(json.dumps({
                        "db_id": db_id,
                        "question": question,
                        "gold_sql": gold_sql,
                        "pred_sql": pred_sql,
                        "status": "missing_db",
                    }) + "\n")
                continue

            total += 1
            if args.limit and total > args.limit:
                break

            if not pred_sql:
                empty_pred += 1
                if out_f:
                    out_f.write(json.dumps({
                        "db_id": db_id,
                        "question": question,
                        "gold_sql": gold_sql,
                        "pred_sql": pred_sql,
                        "status": "empty_pred",
                    }) + "\n")
                continue

            try:
                con = _connect(db_path)
            except Exception as e:
                missing_db += 1
                if out_f:
                    out_f.write(json.dumps({
                        "db_id": db_id,
                        "question": question,
                        "gold_sql": gold_sql,
                        "pred_sql": pred_sql,
                        "status": "db_connect_error",
                        "error": str(e),
                    }) + "\n")
                continue

            status = "mismatch"
            detail: Optional[str] = None
            try:
                try:
                    gold_cols, gold_rows = _run_query(con, gold_sql)
                except Exception as e:
                    gold_error += 1
                    status = "gold_error"
                    detail = str(e)
                    continue

                try:
                    pred_cols, pred_rows = _run_query(con, pred_sql)
                except Exception as e:
                    pred_error += 1
                    status = "pred_error"
                    detail = str(e)
                    continue

                if results_equivalent(gold_cols, gold_rows, pred_cols, pred_rows, gold_sql, pred_sql):
                    exec_correct += 1
                    status = "match"
            finally:
                con.close()
                if out_f:
                    out_f.write(json.dumps({
                        "db_id": db_id,
                        "question": question,
                        "gold_sql": gold_sql,
                        "pred_sql": pred_sql,
                        "status": status,
                        "error": detail,
                    }) + "\n")

    if out_f:
        out_f.close()

    acc = (exec_correct / total) * 100 if total else 0.0
    print(args.preds)
    print(f"Execution Accuracy: {exec_correct}/{total} = {acc:.2f}%")
    print(f"Empty pred SQL: {empty_pred}")
    print(f"Pred SQL errors: {pred_error}")
    print(f"Gold SQL errors: {gold_error}")
    print(f"Missing DB files: {missing_db}")
    if args.out_eval:
        print(f"Wrote per-example results to: {args.out_eval}")

if __name__ == "__main__":
    main()
