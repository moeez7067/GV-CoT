import io
import json
import os

from openai import OpenAI

ROOT           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH     = os.path.join(ROOT, "data", "hotpotqa_train.json")
CHAINS_PATH    = os.path.join(ROOT, "data", "cot_chains.json")
BATCH_ID_PATH  = os.path.join(ROOT, "data", "openai_batch_id.txt")

SLICE_START = 50_026
SLICE_END   = 70_026
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

    print(f"Loading HotpotQA from {INPUT_PATH} ...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    samples = raw[SLICE_START:SLICE_END]
    print(f"  {len(samples)} questions in range (indices {SLICE_START}–{SLICE_END}).")

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
        print(f"Check status or download results using the OpenAI API.")
        return

    print(f"\nBuilding JSONL ({len(pending)} requests) ...")
    lines = []
    for sample in pending:
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

    print("Uploading batch input file ...")
    batch_file = client.files.create(
        file=("batch_input.jsonl", io.BytesIO(jsonl_content.encode("utf-8"))),
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
                print(f"  line={err.line}  code={err.code}  param={err.param}  message={err.message}")
        else:
            print("  No error details returned by API.")
        return

    with open(BATCH_ID_PATH, "w", encoding="utf-8") as f:
        f.write(batch.id)
    print(f"  Batch ID saved to {BATCH_ID_PATH}")
    print(f"\nBatch submitted successfully. {len(pending)} requests queued.")


if __name__ == "__main__":
    main()
