import json
import os

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "cot_chains.json")

with open(PATH, encoding="utf-8") as f:
    data = json.load(f)

before = len(data)

seen = set()
deduped = []
for record in data:
    q = record["question"]
    if q not in seen:
        seen.add(q)
        deduped.append(record)

after = len(deduped)

print(f"Before : {before:,}")
print(f"After  : {after:,}")
print(f"Removed: {before - after:,}")

with open(PATH, "w", encoding="utf-8") as f:
    json.dump(deduped, f, ensure_ascii=False)

print(f"Saved to {PATH}")
