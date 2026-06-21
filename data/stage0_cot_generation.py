import json
import os
import time

from groq import Groq

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH  = os.path.join(ROOT, "data", "hotpotqa_train.json")
OUTPUT_PATH = os.path.join(ROOT, "data", "cot_chains.json")

SLICE_START = 1012
SLICE_END   = 90_447
DELAY       = 0.1        # seconds between API calls
LOG_EVERY   = 100
MODEL       = "llama-3.1-8b-instant"
PROMPT_TMPL = (
    "Answer step by step. Number each step clearly.\n"
    "Question: {question}\n"
    "Provide step by step reasoning then final answer:"
)


def load_existing(path: str) -> dict:
    """Return {id: record} for already-processed questions."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {r["id"]: r for r in records}


def save_all(path: str, records: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(records.values()), f, ensure_ascii=False, indent=2)


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set.")

    client = Groq(api_key=api_key)

    print(f"Loading HotpotQA from {INPUT_PATH} ...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    samples = raw[SLICE_START:SLICE_END]
    print(f"  {len(samples)} questions to process (indices {SLICE_START}–{SLICE_END}).")

    existing = load_existing(OUTPUT_PATH)
    print(f"  {len(existing)} already processed — will skip those.")

    processed = 0
    skipped   = 0
    errors    = 0

    for sample in samples:
        qid      = sample["id"]
        question = sample["question"]
        answer   = sample["answer"]

        if qid in existing:
            skipped += 1
            continue

        prompt = PROMPT_TMPL.format(question=question)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            cot_chain = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [ERROR] id={qid}: {e}")
            errors += 1
            time.sleep(DELAY)
            continue

        existing[qid] = {
            "id":        qid,
            "question":  question,
            "cot_chain": cot_chain,
            "answer":    answer,
        }
        processed += 1

        if processed % LOG_EVERY == 0:
            print(f"  Progress: {len(existing)} total done  "
                  f"(+{processed} this run, {errors} errors, {skipped} skipped)")
            save_all(OUTPUT_PATH, existing)

        time.sleep(DELAY)

    save_all(OUTPUT_PATH, existing)
    print(f"\nDone. Total saved: {len(existing)}  |  "
          f"New this run: {processed}  |  Errors: {errors}  |  Skipped: {skipped}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
