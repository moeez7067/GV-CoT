import json
import os

from groq import Groq

ROOT           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUSIQUE_PATH   = os.path.join(ROOT, "data", "musique_train.json")
CHAINS_PATH    = os.path.join(ROOT, "data", "musique_cot_chains.json")
BATCH_ID_PATH  = os.path.join(ROOT, "data", "musique_groq_batch_id.txt")
MODEL          = "llama-3.3-70b-versatile"


def load_existing(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {r["id"]: r for r in records}


def save_all(path: str, records: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(records.values()), f, ensure_ascii=False, indent=2)


def build_question_lookup(musique_path: str) -> dict:
    with open(musique_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {s["id"]: s for s in raw if s["answerable"]}


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set.")

    client = Groq(api_key=api_key)

    if not os.path.exists(BATCH_ID_PATH):
        raise FileNotFoundError(
            f"No batch ID found at {BATCH_ID_PATH}. "
            "Run stage0_musique_groq_batch.py first."
        )

    with open(BATCH_ID_PATH, "r") as f:
        batch_id = f.read().strip()

    print(f"Checking batch: {batch_id} ...")
    batch = client.batches.retrieve(batch_id)
    print(f"  Status    : {batch.status}")
    print(f"  Total     : {batch.request_counts.total}")
    print(f"  Completed : {batch.request_counts.completed}")
    print(f"  Failed    : {batch.request_counts.failed}")

    if batch.status == "failed":
        print("[ERROR] Batch failed.")
        if batch.errors and batch.errors.data:
            for err in batch.errors.data:
                print(f"  {err.code}: {err.message}")
        return

    if batch.status != "completed":
        print("Not yet complete — run again later.")
        return

    print("Downloading results ...")
    content = client.files.content(batch.output_file_id)
    text = content.read().decode("utf-8")

    question_lookup = build_question_lookup(MUSIQUE_PATH)
    existing        = load_existing(CHAINS_PATH)

    added  = 0
    failed = 0

    for line in text.strip().splitlines():
        if not line.strip():
            continue
        result        = json.loads(line)
        qid           = result.get("custom_id")
        response_body = result.get("response", {}).get("body", {})
        error         = result.get("error")

        if error or not response_body:
            print(f"  [FAILED] id={qid}: {error}")
            failed += 1
            continue

        choices = response_body.get("choices", [])
        if not choices:
            failed += 1
            continue

        cot_chain = choices[0]["message"]["content"].strip()
        sample    = question_lookup.get(qid, {})

        existing[qid] = {
            "id"        : qid,
            "question"  : sample.get("question", ""),
            "cot_chain" : cot_chain,
            "answer"    : sample.get("answer", ""),
            "model_name": MODEL,
            "source_api": "groq_batch",
        }
        added += 1

    if added > 0:
        save_all(CHAINS_PATH, existing)

    print(f"\n=== Summary ===")
    print(f"  Added this run : {added}")
    print(f"  Failed         : {failed}")
    print(f"  Total in file  : {len(existing)}")
    if added > 0:
        print(f"  Output         : {CHAINS_PATH}")


if __name__ == "__main__":
    main()
