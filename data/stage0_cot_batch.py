import io
import json
import os

from groq import Groq

ROOT            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH      = os.path.join(ROOT, "data", "hotpotqa_train.json")
CHAINS_PATH     = os.path.join(ROOT, "data", "cot_chains.json")
BATCH_ID_1_PATH = os.path.join(ROOT, "data", "batch_id_1.txt")
BATCH_ID_2_PATH = os.path.join(ROOT, "data", "batch_id_2.txt")

SLICE_START = 1012
SLICE_END   = 90_447
BATCH_SIZE  = 45_000
MODEL       = "llama-3.1-8b-instant"
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


def submit_batch(client, chunk: list, batch_num: int, batch_id_path: str) -> None:
    print(f"\nBuilding JSONL for batch {batch_num} ({len(chunk)} requests) ...")
    lines = []
    for sample in chunk:
        lines.append(json.dumps({
            "custom_id": sample["id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": PROMPT_TMPL.format(question=sample["question"])}
                ],
            },
        }))
    jsonl_content = "\n".join(lines) + "\n"

    print(f"Uploading batch {batch_num} input file ...")
    batch_file = client.files.create(
        file=("batch_input.jsonl", io.BytesIO(jsonl_content.encode("utf-8"))),
        purpose="batch",
    )
    print(f"  File uploaded: id={batch_file.id}")

    print(f"Creating batch {batch_num} job ...")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  Batch created: id={batch.id}  status={batch.status}")

    if batch.status == "failed":
        print(f"\n[ERROR] Batch {batch_num} failed immediately during validation.")
        if batch.errors and batch.errors.data:
            for err in batch.errors.data:
                print(f"  line={err.line}  code={err.code}  param={err.param}  message={err.message}")
        else:
            print("  No error details returned by API.")
        return

    with open(batch_id_path, "w", encoding="utf-8") as f:
        f.write(batch.id)
    print(f"  Batch ID saved to {batch_id_path}")
    print(f"  Batch {batch_num} submitted successfully. {len(chunk)} requests queued.")


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set.")

    client = Groq(api_key=api_key)

    print(f"Loading HotpotQA from {INPUT_PATH} ...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    samples = raw[SLICE_START:SLICE_END]
    print(f"  {len(samples)} questions in range (indices {SLICE_START}–{SLICE_END}).")

    processed_ids = load_processed_ids(CHAINS_PATH)
    print(f"  {len(processed_ids)} already processed — will skip those.")

    pending = [s for s in samples if s["id"] not in processed_ids]
    print(f"  {len(pending)} questions pending.")

    if not pending:
        print("Nothing to do — all questions already processed.")
        return

    batch1_exists = os.path.exists(BATCH_ID_1_PATH)
    batch2_exists = os.path.exists(BATCH_ID_2_PATH)

    if batch1_exists and batch2_exists:
        with open(BATCH_ID_1_PATH) as f:
            id1 = f.read().strip()
        with open(BATCH_ID_2_PATH) as f:
            id2 = f.read().strip()
        print(f"\nBoth batches already submitted.")
        print(f"  Batch 1: {id1}")
        print(f"  Batch 2: {id2}")
        print("Run stage0_batch_download.py to check status and download results.")
        return

    if not batch1_exists:
        chunk = pending[:BATCH_SIZE]
        print(f"\nSubmitting batch 1: first {len(chunk)} of {len(pending)} pending questions.")
        print("  NOTE: submit batch 2 before merging results so the pending list stays consistent.")
        submit_batch(client, chunk, 1, BATCH_ID_1_PATH)
    else:
        chunk = pending[BATCH_SIZE:]
        if not chunk:
            print("\n[WARNING] No items remain beyond BATCH_SIZE in the pending list.")
            print("  Batch 1 results may have already been merged — check cot_chains.json.")
            return
        print(f"\nSubmitting batch 2: remaining {len(chunk)} questions.")
        submit_batch(client, chunk, 2, BATCH_ID_2_PATH)

    print(f"\nRun stage0_batch_download.py to check status and download results.")


if __name__ == "__main__":
    main()
