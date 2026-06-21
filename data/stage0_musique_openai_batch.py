import io
import json
import os

from openai import OpenAI

ROOT           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUSIQUE_PATH   = os.path.join(ROOT, "data", "musique_train.json")
CHAINS_PATH    = os.path.join(ROOT, "data", "musique_cot_chains.json")
BATCH_ID_PATH  = os.path.join(ROOT, "data", "musique_openai_batch_id.txt")

SLICE_START = 0
SLICE_END   = 12_000
MODEL       = "gpt-4.1-nano"
PROMPT_TMPL = (
    "Answer step by step. Number each step clearly.\n"
    "Question: {question}\n"
    "Provide step by step reasoning then final answer:"
)


def load_processed_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {r["id"] for r in records}


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)

    print(f"Loading MuSiQue from {MUSIQUE_PATH} ...")
    with open(MUSIQUE_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Filter answerable only then slice
    answerable = [r for r in raw if r["answerable"]]
    samples    = answerable[SLICE_START:SLICE_END]
    print(f"  {len(samples)} answerable questions (indices {SLICE_START}–{SLICE_END}).")

    processed_ids = load_processed_ids(CHAINS_PATH)
    print(f"  {len(processed_ids)} already processed — will skip those.")

    pending = [s for s in samples if s["id"] not in processed_ids]
    print(f"  {len(pending)} questions to submit in batch.")

    if not pending:
        print("Nothing to do — all questions already processed.")
        return

    if os.path.exists(BATCH_ID_PATH):
        with open(BATCH_ID_PATH) as f:
            existing_id = f.read().strip()
        print(f"\nBatch already submitted: {existing_id}")
        print("Run stage0_musique_openai_batch_download.py to check status.")
        return

    print(f"\nBuilding JSONL ({len(pending)} requests) ...")
    lines = []
    for sample in pending:
        lines.append(json.dumps({
            "custom_id": sample["id"],
            "method":    "POST",
            "url":       "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": PROMPT_TMPL.format(question=sample["question"])}
                ],
            },
        }))
    jsonl_content = "\n".join(lines) + "\n"

    print("Uploading batch input file ...")
    batch_file = client.files.create(
        file=("musique_batch_openai.jsonl", io.BytesIO(jsonl_content.encode("utf-8"))),
        purpose="batch",
    )
    print(f"  File uploaded: id={batch_file.id}")

    print("Creating batch job ...")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  Batch created: id={batch.id}  status={batch.status}")

    if batch.status == "failed":
        print("\n[ERROR] Batch failed immediately during validation.")
        if batch.errors and batch.errors.data:
            for err in batch.errors.data:
                print(f"  line={err.line}  code={err.code}  message={err.message}")
        return

    with open(BATCH_ID_PATH, "w", encoding="utf-8") as f:
        f.write(batch.id)
    print(f"  Batch ID saved to {BATCH_ID_PATH}")
    print(f"\nBatch submitted. {len(pending)} requests queued (24h window).")
    print("Run stage0_musique_openai_batch_download.py when complete.")


if __name__ == "__main__":
    main()
