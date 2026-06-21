import json
import os

import anthropic

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAINS_PATH = os.path.join(ROOT, "data", "cot_chains.json")
INPUT_PATH  = os.path.join(ROOT, "data", "hotpotqa_train.json")

BATCH_ID_FILES = [
    (1, os.path.join(ROOT, "data", "anthropic_batch_id_1.txt")),
    (2, os.path.join(ROOT, "data", "anthropic_batch_id_2.txt")),
    (3, os.path.join(ROOT, "data", "anthropic_batch_id_3.txt")),
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


def process_batch(client, batch_id: str, batch_num: int,
                  question_lookup: dict, existing: dict) -> tuple:
    """Check one batch and merge results. Returns (added, errored, still_running)."""
    print(f"\n--- Batch {batch_num}: {batch_id} ---")
    batch = client.messages.batches.retrieve(batch_id)
    print(f"  Processing status : {batch.processing_status}")
    print(f"  Request counts    : {batch.request_counts}")

    if batch.processing_status != "ended":
        print("  Not yet complete — run again later.")
        return 0, 0, True

    print("  Downloading results ...")
    added = 0
    errored = 0

    for result in client.messages.batches.results(batch_id):
        qid = result.custom_id

        if result.result.type == "succeeded":
            content_blocks = result.result.message.content
            if not content_blocks:
                errored += 1
                continue
            cot_chain = content_blocks[0].text.strip()
            sample = question_lookup.get(qid, {})
            existing[qid] = {
                "id":        qid,
                "question":  sample.get("question", ""),
                "cot_chain": cot_chain,
                "answer":    sample.get("answer", ""),
            }
            added += 1
        else:
            print(f"    [{result.result.type.upper()}] id={qid}")
            errored += 1

    print(f"  Added: {added}  Errored/expired: {errored}")
    return added, errored, False


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set.")

    client = anthropic.Anthropic(api_key=api_key)

    present = [(num, path) for num, path in BATCH_ID_FILES if os.path.exists(path)]
    if not present:
        raise FileNotFoundError(
            "No batch ID files found. Expected anthropic_batch_id_1.txt, "
            "anthropic_batch_id_2.txt, and/or anthropic_batch_id_3.txt"
        )

    question_lookup = build_question_lookup(INPUT_PATH)
    existing = load_existing(CHAINS_PATH)

    total_added = 0
    total_errored = 0
    incomplete = 0

    for batch_num, id_path in present:
        with open(id_path, "r", encoding="utf-8") as f:
            batch_id = f.read().strip()
        added, errored, still_running = process_batch(
            client, batch_id, batch_num, question_lookup, existing
        )
        total_added += added
        total_errored += errored
        if still_running:
            incomplete += 1

    if total_added > 0:
        save_all(CHAINS_PATH, existing)

    print(f"\n=== Summary ===")
    print(f"  Batches checked      : {len(present)}")
    print(f"  Still in progress    : {incomplete}")
    print(f"  Added this run       : {total_added}")
    print(f"  Errored / expired    : {total_errored}")
    print(f"  Total in file        : {len(existing)}")
    if total_added > 0:
        print(f"  Output               : {CHAINS_PATH}")


if __name__ == "__main__":
    main()
