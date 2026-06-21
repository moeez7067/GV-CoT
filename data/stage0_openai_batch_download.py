import json
import os

from openai import OpenAI

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAINS_PATH = os.path.join(ROOT, "data", "cot_chains.json")
INPUT_PATH  = os.path.join(ROOT, "data", "hotpotqa_train.json")

BATCH_ID_FILES = [
    (1, os.path.join(ROOT, "data", "openai_batch_id.txt")),
    (2, os.path.join(ROOT, "data", "openai_batch2_id.txt")),
]


def load_existing(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {r["id"]: r for r in records}


def save_all(path: str, records: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(records.values()), f, ensure_ascii=False, indent=2)


def build_question_lookup(input_path: str) -> dict:
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {s["id"]: s for s in raw}


def print_batch_errors(client, batch) -> None:
    if batch.errors and batch.errors.data:
        for err in batch.errors.data:
            print(f"  line={err.line}  code={err.code}  param={err.param}  message={err.message}")
    else:
        print("  No structured errors returned.")
    if batch.error_file_id:
        print(f"  Error file id: {batch.error_file_id}")
        try:
            error_content = client.files.content(batch.error_file_id)
            print("  Error file contents:")
            for line in error_content.text.strip().splitlines():
                print(f"    {line}")
        except Exception as e:
            print(f"  Could not download error file: {e}")


def process_batch(client, batch_id: str, batch_num: int,
                  question_lookup: dict, existing: dict) -> tuple:
    """Check one batch and merge results. Returns (added, failed, still_running)."""
    print(f"\n--- Batch {batch_num}: {batch_id} ---")
    batch = client.batches.retrieve(batch_id)
    print(f"  Status    : {batch.status}")
    print(f"  Total     : {batch.request_counts.total}")
    print(f"  Completed : {batch.request_counts.completed}")
    print(f"  Failed    : {batch.request_counts.failed}")

    if batch.status == "failed":
        print(f"  [ERROR] Batch {batch_num} failed.")
        print_batch_errors(client, batch)
        return 0, 0, False

    if batch.status != "completed":
        print("  Not yet complete — run again later.")
        return 0, 0, True

    print("  Downloading results ...")
    content = client.files.content(batch.output_file_id)

    added = 0
    failed = 0
    for line in content.text.strip().splitlines():
        if not line.strip():
            continue
        result = json.loads(line)
        qid = result.get("custom_id")
        response_body = result.get("response", {}).get("body", {})
        error = result.get("error")

        if error or not response_body:
            print(f"    [FAILED] id={qid}: {error}")
            failed += 1
            continue

        choices = response_body.get("choices", [])
        if not choices:
            failed += 1
            continue

        cot_chain = choices[0]["message"]["content"].strip()
        sample = question_lookup.get(qid, {})
        existing[qid] = {
            "id":        qid,
            "question":  sample.get("question", ""),
            "cot_chain": cot_chain,
            "answer":    sample.get("answer", ""),
        }
        added += 1

    print(f"  Added: {added}  Failed: {failed}")
    return added, failed, False


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)

    present = [(num, path) for num, path in BATCH_ID_FILES if os.path.exists(path)]
    if not present:
        raise FileNotFoundError(
            "No batch ID files found. Expected openai_batch_id.txt and/or openai_batch2_id.txt"
        )

    question_lookup = build_question_lookup(INPUT_PATH)
    existing = load_existing(CHAINS_PATH)

    total_added = 0
    total_failed = 0
    incomplete = 0

    for batch_num, id_path in present:
        with open(id_path, "r", encoding="utf-8") as f:
            batch_id = f.read().strip()
        added, failed, still_running = process_batch(
            client, batch_id, batch_num, question_lookup, existing
        )
        total_added += added
        total_failed += failed
        if still_running:
            incomplete += 1

    if total_added > 0:
        save_all(CHAINS_PATH, existing)

    print(f"\n=== Summary ===")
    print(f"  Batches checked    : {len(present)}")
    print(f"  Still in progress  : {incomplete}")
    print(f"  Added this run     : {total_added}")
    print(f"  Failed requests    : {total_failed}")
    print(f"  Total in file      : {len(existing)}")
    if total_added > 0:
        print(f"  Output             : {CHAINS_PATH}")


if __name__ == "__main__":
    main()
