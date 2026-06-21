import torch  # MUST be first import — Windows DLL fix

import json
import os
import sys
import time
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="MuSiQue Stage 0 — CoT Generation")
parser.add_argument("--api",   required=True, choices=["openai", "groq"])
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end",   type=int, default=19938)
args = parser.parse_args()

BASE_DIR     = Path(r"D:\gvcot")
MUSIQUE_JSON = BASE_DIR / "data" / "musique_train.json"
OUT_JSON     = BASE_DIR / "data" / "musique_cot_chains.json"
CHECKPOINT   = BASE_DIR / "data" / f"musique_cot_checkpoint_{args.api}.json"

if args.api == "openai":
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    MODEL  = "gpt-4.1-nano"
    RPM    = 500
    print(f"[Stage 0] Using OpenAI — model: {MODEL}")
elif args.api == "groq":
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    MODEL  = "llama-3.3-70b-versatile"
    RPM    = 900
    print(f"[Stage 0] Using Groq — model: {MODEL}")

SLEEP = 60.0 / RPM

print(f"[Stage 0] Loading {MUSIQUE_JSON} ...")
with open(MUSIQUE_JSON, encoding="utf-8") as f:
    data = json.load(f)

answerable = [r for r in data if r["answerable"]]
subset     = answerable[args.start:args.end]
print(f"[Stage 0] Processing {len(subset):,} questions (indices {args.start}–{args.end-1})")

done_ids = set()
results  = []
if CHECKPOINT.exists():
    with open(CHECKPOINT, encoding="utf-8") as f:
        results = json.load(f)
    done_ids = {r["id"] for r in results}
    print(f"[Stage 0] Resuming — {len(done_ids):,} already done")

SYSTEM_PROMPT = (
    "You are a careful reasoning assistant. "
    "Given a multi-hop question, reason through it step by step. "
    "Each step should be a single clear reasoning action. "
    "End with a line starting with '# Final Answer' followed by the answer."
)

def make_prompt(question):
    return f"Question: {question}\n\nPlease reason through this step by step to find the answer."

errors = 0
for i, record in enumerate(subset):
    qid = record["id"]
    if qid in done_ids:
        continue
    if (i + 1) % 100 == 0:
        print(f"  ... {i+1:,}/{len(subset):,} | done: {len(results):,} | errors: {errors}")
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": make_prompt(record["question"])},
            ],
            max_tokens=512,
            temperature=0.2,
        )
        cot_chain = response.choices[0].message.content.strip()
        results.append({
            "id"        : qid,
            "question"  : record["question"],
            "answer"    : record["answer"],
            "cot_chain" : cot_chain,
            "model_name": MODEL,
            "source_api": args.api,
            "answerable": True,
        })
        done_ids.add(qid)
    except Exception as e:
        errors += 1
        print(f"  [ERROR] {qid}: {e}")
        time.sleep(5)
        continue
    time.sleep(SLEEP)
    if len(results) % 200 == 0:
        with open(CHECKPOINT, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)

with open(CHECKPOINT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False)
print(f"[Stage 0] Checkpoint saved: {len(results):,} records")

existing = []
if OUT_JSON.exists():
    with open(OUT_JSON, encoding="utf-8") as f:
        existing = json.load(f)
existing_ids = {r["id"] for r in existing}
new_records  = [r for r in results if r["id"] not in existing_ids]
merged       = existing + new_records
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)
print(f"[Stage 0] Saved {len(merged):,} total records to {OUT_JSON}")
print(f"[Stage 0] COMPLETE")
