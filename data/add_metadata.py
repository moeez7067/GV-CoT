import json
import os

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOTPOT_PATH = os.path.join(ROOT, "data", "hotpotqa_train.json")
COT_PATH    = os.path.join(ROOT, "data", "cot_chains.json")

print(f"Loading {HOTPOT_PATH} ...")
with open(HOTPOT_PATH, encoding="utf-8") as f:
    hotpot = json.load(f)

anthropic_ids  = {s["id"] for s in hotpot[0:27_000]}
gpt_batch1_ids = {s["id"] for s in hotpot[50_026:70_026]}
gpt_batch2_ids = {s["id"] for s in hotpot[77_526:90_447]}
gpt_ids        = gpt_batch1_ids | gpt_batch2_ids

print(f"  anthropic_ids : {len(anthropic_ids):,}")
print(f"  gpt_batch1_ids: {len(gpt_batch1_ids):,}")
print(f"  gpt_batch2_ids: {len(gpt_batch2_ids):,}")

print(f"\nLoading {COT_PATH} ...")
with open(COT_PATH, encoding="utf-8") as f:
    data = json.load(f)
print(f"  {len(data):,} records loaded.")

for record in data:
    rid = record["id"]
    if rid in anthropic_ids:
        record["model_name"] = "claude-haiku-4-5"
    elif rid in gpt_ids:
        record["model_name"] = "gpt-4.1-nano"
    else:
        record["model_name"] = "llama-3.1-8b-instant"
    record["prompt_version"] = "v1"
    record["temperature"]    = 1.0

counts = {}
for record in data:
    m = record["model_name"]
    counts[m] = counts.get(m, 0) + 1

print("\nRecords per model:")
for model, count in sorted(counts.items()):
    print(f"  {model}: {count:,}")

with open(COT_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

print(f"\nSaved to {COT_PATH}")
