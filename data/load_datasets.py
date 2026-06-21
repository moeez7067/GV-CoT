import json
import os
import sys

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' library not found. Install it with:")
    print("  pip install datasets")
    sys.exit(1)

# Ensure stdout handles Unicode on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def save_dataset(dataset, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(dataset), f, ensure_ascii=False, indent=2)
    print(f"Saved {len(dataset)} records to {path}")


def main():
    # --- HotpotQA distractor ---
    print("Loading HotpotQA (distractor setting)...")
    hotpotqa = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    print(f"HotpotQA train size: {len(hotpotqa)}")
    print("\nFirst HotpotQA sample:")
    print(json.dumps(hotpotqa[0], indent=2, ensure_ascii=False))

    hotpotqa_path = os.path.join(DATA_DIR, "hotpotqa_train.json")
    save_dataset(hotpotqa, hotpotqa_path)

    # --- MuSiQue ---
    print("\nLoading MuSiQue...")
    musique = load_dataset("bdsaglam/musique", split="train")
    print(f"MuSiQue train size: {len(musique)}")
    print("\nFirst MuSiQue sample:")
    print(json.dumps(musique[0], indent=2, ensure_ascii=False))

    musique_path = os.path.join(DATA_DIR, "musique_train.json")
    save_dataset(musique, musique_path)

    print("\nAll done.")


if __name__ == "__main__":
    main()
