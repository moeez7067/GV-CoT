import json
import os

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(ROOT, "data", "hotpotqa_train.json")

BATCH_CONFIGS = [
    (1,  0,      9_000,  os.path.join(ROOT, "data", "anthropic_batch_id_1.txt")),
    (2,  9_000,  18_000, os.path.join(ROOT, "data", "anthropic_batch_id_2.txt")),
    (3,  18_000, 27_000, os.path.join(ROOT, "data", "anthropic_batch_id_3.txt")),
]

MODEL      = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024
PROMPT_TMPL = (
    "Answer step by step. Number each step clearly.\n"
    "Question: {question}\n"
    "Provide step by step reasoning then final answer:"
)


def submit_batch(client, batch_num: int, start: int, end: int,
                 batch_id_path: str, raw: list) -> None:
    samples = raw[start:end]
    print(f"\nBuilding batch {batch_num}: {len(samples)} requests "
          f"(raw[{start}:{end}]) ...")

    requests = [
        Request(
            custom_id=sample["id"],
            params=MessageCreateParamsNonStreaming(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{
                    "role": "user",
                    "content": PROMPT_TMPL.format(question=sample["question"]),
                }],
            ),
        )
        for sample in samples
    ]

    print(f"Submitting batch {batch_num} to Anthropic Message Batches API ...")
    batch = client.messages.batches.create(requests=requests)
    print(f"  Batch created: id={batch.id}  status={batch.processing_status}")

    with open(batch_id_path, "w", encoding="utf-8") as f:
        f.write(batch.id)
    print(f"  Batch ID saved to {batch_id_path}")
    print(f"  Batch {batch_num} submitted successfully. {len(samples)} requests queued.")


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set.")

    client = anthropic.Anthropic(api_key=api_key)

    print(f"Loading HotpotQA from {INPUT_PATH} ...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    print(f"  {len(raw)} total questions loaded.")

    for batch_num, start, end, batch_id_path in BATCH_CONFIGS:
        if not os.path.exists(batch_id_path):
            submit_batch(client, batch_num, start, end, batch_id_path, raw)
            remaining = sum(
                1 for _, _, _, p in BATCH_CONFIGS if not os.path.exists(p)
            ) - 1
            if remaining > 0:
                print(f"\nRun again to submit the next batch "
                      f"({remaining} remaining).")
            return

    print("\nAll 3 batches already submitted:")
    for batch_num, start, end, batch_id_path in BATCH_CONFIGS:
        with open(batch_id_path) as f:
            batch_id = f.read().strip()
        print(f"  Batch {batch_num} (raw[{start}:{end}]): {batch_id}")
    print("Create a download script to check status and retrieve results.")


if __name__ == "__main__":
    main()
