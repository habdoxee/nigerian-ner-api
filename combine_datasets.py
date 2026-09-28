"""
Merge your original clean MasakhaNER training data with the new large-scale
silver-labeled NaijaSenti tweet data, producing one combined dataset ready
for continued fine-tuning (code-mixing / noisy-text adaptation).

Design choice: the original clean validation and test splits are kept
UNCHANGED (not mixed with silver data) -- this preserves your existing,
trustworthy clean-text benchmark (F1 0.84) for direct before/after
comparison. Only the TRAIN split gets the new silver data added.

Usage:
    python combine_datasets.py

Requires:
    pip install datasets

Output:
    ./combined_ner_dataset_v2/  (a new DatasetDict, same structure as before)
"""

import os
from datasets import Dataset, DatasetDict, Features, Sequence, ClassLabel, Value, load_from_disk

ORIGINAL_DATASET_DIR = "./combined_ner_dataset"
SILVER_DATA_DIR = "./silver_training_data"
OUTPUT_DIR = "./combined_ner_dataset_v2"

LABEL_LIST = [
    "B-DATE", "B-LOC", "B-ORG", "B-PER",
    "I-DATE", "I-LOC", "I-ORG", "I-PER",
    "O",
]


def read_conll(filepath):
    """Same reader as prepare_dataset.py -- kept identical so both scripts
    produce compatible data."""
    sentences = []
    tokens, tags = [], []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                if tokens:
                    sentences.append({"tokens": tokens, "ner_tags": tags})
                    tokens, tags = [], []
            else:
                splits = line.split()
                if len(splits) >= 2:
                    tokens.append(splits[0])
                    tags.append(splits[-1])
        if tokens:
            sentences.append({"tokens": tokens, "ner_tags": tags})
    return sentences


print(f"Loading original dataset from {ORIGINAL_DATASET_DIR} ...")
original = load_from_disk(ORIGINAL_DATASET_DIR)
print(original)

print(f"\nLoading silver training data from {SILVER_DATA_DIR} ...")
silver_examples = []
for fname in sorted(os.listdir(SILVER_DATA_DIR)):
    if not fname.endswith(".conll"):
        continue
    filepath = os.path.join(SILVER_DATA_DIR, fname)
    examples = read_conll(filepath)
    print(f"  {fname}: {len(examples)} sentences")
    silver_examples.extend(examples)

print(f"\nTotal silver examples: {len(silver_examples)}")

# Convert original train split back to raw tokens/string tags so it can be
# concatenated with the silver examples before re-encoding as ClassLabel.
label_names = original["train"].features["ner_tags"].feature.names
original_train_raw = [
    {
        "tokens": ex["tokens"],
        "ner_tags": [label_names[i] for i in ex["ner_tags"]],
    }
    for ex in original["train"]
]

combined_train_raw = original_train_raw + silver_examples
print(f"\nCombined train size: {len(original_train_raw)} original + "
      f"{len(silver_examples)} silver = {len(combined_train_raw)} total")

features = Features({
    "tokens": Sequence(Value("string")),
    "ner_tags": Sequence(ClassLabel(names=LABEL_LIST)),
})

combined_train_dataset = Dataset.from_list(combined_train_raw, features=features)

# Validation and test stay exactly as they were -- untouched clean benchmark.
final_dataset = DatasetDict({
    "train": combined_train_dataset,
    "validation": original["validation"],
    "test": original["test"],
})

print("\nFinal combined dataset:")
print(final_dataset)

final_dataset.save_to_disk(OUTPUT_DIR)
print(f"\nSaved combined dataset to {OUTPUT_DIR}")
print("Update your training script's DATASET_DIR to point here for the")
print("code-mixing adaptation fine-tuning run.")
