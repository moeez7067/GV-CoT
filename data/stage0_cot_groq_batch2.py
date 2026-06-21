import io
import json
import os

from groq import Groq

ROOT           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH     = os.path.join(ROOT, "data", "hotpotqa_train.json")
BATCH_ID_PATH  = os.path.join(ROOT, "data", "groq_batch2_id.txt")

SLICE_START = 70_026
SLICE_END   = 77_526
MODEL       = "llama-3.1-8b-instant"
PROMPT_TMPL = (
    "Answer step by step. Number each step clearly.\n"
    "Question: {question}\n"
    "Provide step by step reasoning then final answer:"
)


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set.")

    client = Groq(api_key=api_key)

    if os.path.exists(BATCH_ID_PATH):
        with open(BATCH_ID_PATH) as f:
            existing_id = f.read().strip()
        print(f"Batch already submitted: {existing_id}")
        print("Run a download script to check status and retrieve results.")
        return

    print(f"Loading HotpotQA from {INPUT_PATH} ...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    samples = raw[SLICE_START:SLICE_END]
    print(f"  {len(samples)} questions to submit (raw[{SLICE_START}:{SLICE_END}]).")

    print(f"Building JSONL ({len(samples)} requests) ...")
    lines = []
    for sample in samples:
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
    print(f"\nBatch submitted successfully. {len(samples)} requests queued.")


if __name__ == "__main__":
    main()
