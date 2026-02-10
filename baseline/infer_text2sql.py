import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_spider_split(spider_dir: str, split: str) -> Tuple[List[dict], Dict[str, dict]]:
    """
    Loads SPIDER split JSON and tables.json from a local SPIDER directory.

    Expected structure:
      spider_dir/
        train_spider.json
        dev.json
        tables.json
        database/<db_id>/<db_id>.sqlite
    """
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


def build_schema_str_from_tables(db_id: str, tables_by_db: Dict[str, dict]) -> str:
    """
    Builds a readable schema string from tables.json for a given db_id.
    """
    if db_id not in tables_by_db:
        raise KeyError(f"db_id not found in tables.json: {db_id}")

    t = tables_by_db[db_id]
    table_names = t["table_names_original"]
    col_names = t["column_names_original"]  # list of [table_id, column_name]
    col_types = t["column_types"]

    by_table: Dict[int, List[str]] = {i: [] for i in range(len(table_names))}
    for (t_id, c_name), c_type in zip(col_names, col_types):
        if t_id == -1:
            continue
        by_table[t_id].append(f"{c_name} ({c_type})")

    lines = []
    for t_id, t_name in enumerate(table_names):
        cols = ", ".join(by_table[t_id])
        lines.append(f"Table {t_name}: {cols}")
    return "\n".join(lines)


def build_prompt(schema: str, question: str, prompt_style: str) -> str:
    if prompt_style == "schema_question_sql":
        return (
            "You are an assistant that writes SQL.\n"
            "Given the database schema, write a SQL query that answers the question.\n\n"
            f"Schema:\n{schema}\n\n"
            f"Question: {question}\n"
            "SQL:"
        )

    if prompt_style == "direct":
        return f"{schema}\n\nQuestion: {question}\nSQL:"

    raise ValueError(f"Unknown prompt_style: {prompt_style}")


@torch.no_grad()
def generate_sql(tok, model, prompt: str, max_new_tokens: int) -> str:
    device = model.device
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        temperature=0.0,
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
            schema = build_schema_str_from_tables(db_id, tables_by_db)
            prompt = build_prompt(schema, ex["question"], args.prompt_style)
            pred_sql = generate_sql(tok, model, prompt, args.max_new_tokens)

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
