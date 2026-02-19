import argparse
import json
import os
import re
from typing import Dict, List, Tuple, Set

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_spider_split(spider_dir: str, split: str) -> Tuple[List[dict], Dict[str, dict]]:
    if split == "train":
        data_path = os.path.join(spider_dir, "train_spider.json")
    elif split == "dev":
        data_path = os.path.join(spider_dir, "dev.json")
    else:
        raise ValueError("split must be 'train' or 'dev'")

    tables_path = os.path.join(spider_dir, "tables.json")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find split file: {data_path}")
    if not os.path.exists(tables_path):
        raise FileNotFoundError(f"Could not find tables file: {tables_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(tables_path, "r", encoding="utf-8") as f:
        tables = json.load(f)

    tables_by_db = {t["db_id"]: t for t in tables}
    return data, tables_by_db


def get_table_names(db_id: str, tables_by_db: Dict[str, dict]) -> List[str]:
    if db_id not in tables_by_db:
        raise KeyError(f"db_id not found in tables.json: {db_id}")
    return tables_by_db[db_id]["table_names_original"]


def build_schema_str_from_tables(db_id: str, tables_by_db: Dict[str, dict]) -> str:
    """
    Builds a readable schema string from tables.json for a given db_id, including PK/FK relations.
    """
    if db_id not in tables_by_db:
        raise KeyError(f"db_id not found in tables.json: {db_id}")

    t = tables_by_db[db_id]
    table_names = t["table_names_original"]
    col_names = t["column_names_original"]  # list of [table_id, column_name]
    col_types = t["column_types"]
    primary_keys = t.get("primary_keys", [])
    foreign_keys = t.get("foreign_keys", [])  # list of [col_idx_1, col_idx_2]

    by_table: Dict[int, List[str]] = {i: [] for i in range(len(table_names))}
    for ((t_id, c_name), c_type) in zip(col_names, col_types):
        if t_id == -1:
            continue
        by_table[t_id].append(f"{c_name} ({c_type})")

    lines: List[str] = []
    for t_id, t_name in enumerate(table_names):
        cols = ", ".join(by_table[t_id])
        lines.append(f"Table {t_name}: {cols}")

    pk_lines: List[str] = []
    for col_idx in primary_keys:
        t_id, c_name = col_names[col_idx]
        if t_id != -1:
            pk_lines.append(f"{table_names[t_id]}.{c_name}")

    fk_lines: List[str] = []
    for c1, c2 in foreign_keys:
        t1, c1n = col_names[c1]
        t2, c2n = col_names[c2]
        if t1 != -1 and t2 != -1:
            fk_lines.append(f"{table_names[t1]}.{c1n} = {table_names[t2]}.{c2n}")

    if pk_lines:
        lines.append("Primary Keys: " + "; ".join(pk_lines))
    if fk_lines:
        lines.append("Foreign Keys: " + "; ".join(fk_lines))

    return "\n".join(lines)


def build_prompt(schema: str, question: str, prompt_style: str, allowed_tables: List[str]) -> str:
    """
    Keep instructions short and avoid prominent tokens like 'SQLite'/'SQL'.
    Add explicit allowed table list to reduce hallucinated tables (e.g., 'Country', 'concert_Name').
    """
    tables_line = "Allowed tables: " + ", ".join(allowed_tables)

    if prompt_style == "schema_question_sql":
        return (
            "Task: Write one query that answers the question using the given schema.\n"
            "Rules:\n"
            "- Use only tables and columns from the schema.\n"
            "- Use only the allowed table names.\n"
            "- Return only the query text.\n"
            f"{tables_line}\n\n"
            f"Schema:\n{schema}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    if prompt_style == "direct":
        return (
            f"{tables_line}\n\n"
            f"{schema}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    raise ValueError(f"Unknown prompt_style: {prompt_style}")


def clean_pred_sql(text: str) -> str:
    if not text:
        return ""

    s = text.strip()
    s = s.replace("```sql", "```").replace("```SQL", "```")
    s = s.strip("`").strip()

    m = re.search(r"\bselect\b", s, flags=re.IGNORECASE)
    if not m:
        return ""

    s = s[m.start():].strip()

    if "```" in s:
        s = s.split("```", 1)[0].strip()

    s = s.strip()
    s = s.rstrip(";").strip()
    return s


def extract_tables_from_sql(sql: str) -> Set[str]:
    """
    Extract table identifiers appearing after FROM/JOIN.
    This is intentionally simple (works well enough for Spider baseline diagnostics).
    """
    if not sql:
        return set()

    s = sql
    # remove quoted strings to avoid confusing FROM 'x'
    s = re.sub(r"'[^']*'", "''", s)
    s = re.sub(r'"[^"]*"', '""', s)

    # capture FROM <name> or JOIN <name>
    found = set()
    for m in re.finditer(r"\b(from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", s, flags=re.IGNORECASE):
        found.add(m.group(2))
    return found


@torch.no_grad()
def generate_sql(tok, model, prompt: str, max_new_tokens: int) -> str:
    device = model.device
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )
    return tok.decode(out[0], skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model name (seq2seq)")
    ap.add_argument("--spider_dir", required=True, help="Path to local SPIDER directory")
    ap.add_argument("--split", default="dev", choices=["train", "dev"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument(
        "--prompt_style",
        default="schema_question_sql",
        choices=["schema_question_sql", "direct"],
    )
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument(
        "--reject_unknown_tables",
        action="store_true",
        help="If set, blank predictions that reference tables not in schema.",
    )
    args = ap.parse_args()

    data, tables_by_db = load_spider_split(args.spider_dir, args.split)
    n = min(args.n, len(data))
    subset = data[:n]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for ex in tqdm(subset, desc=f"Running {args.model}"):
            db_id = ex["db_id"]
            allowed_tables = get_table_names(db_id, tables_by_db)
            allowed_set = set(allowed_tables)

            schema = build_schema_str_from_tables(db_id, tables_by_db)
            prompt = build_prompt(schema, ex["question"], args.prompt_style, allowed_tables)

            pred = generate_sql(tok, model, prompt, args.max_new_tokens)
            pred_sql = clean_pred_sql(pred)

            if args.reject_unknown_tables and pred_sql:
                used = extract_tables_from_sql(pred_sql)
                unknown = {t for t in used if t not in allowed_set}
                if unknown:
                    # Blank it out so eval_exec counts it as empty_pred (cleaner diagnostics)
                    pred_sql = ""

            rec = {
                "model": args.model,
                "db_id": db_id,
                "question": ex["question"],
                "gold_sql": ex.get("query", ""),
                "pred_sql": pred_sql,
                "prompt_style": args.prompt_style,
            }
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
