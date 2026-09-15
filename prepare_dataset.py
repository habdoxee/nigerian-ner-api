from datasets import Dataset, DatasetDict, Features, Sequence, ClassLabel, Value
import os

# Languages to combine
LANGUAGES = ["pcm", "hau", "ibo", "yor"]
DATA_DIR = "data"  # run this script from inside masakhane-ner-main\masakhane-ner-main

def read_conll(filepath):
    """Reads a CoNLL-format file and returns list of {tokens, ner_tags}"""
    sentences = []
    tokens = []
    tags = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                if tokens:
                    sentences.append({"tokens": tokens, "ner_tags": tags})
                    tokens = []
                    tags = []
            else:
                splits = line.split()
                if len(splits) >= 2:
                    tokens.append(splits[0])
                    tags.append(splits[-1])
        if tokens:  # catch last sentence if no trailing blank line
            sentences.append({"tokens": tokens, "ner_tags": tags})
    return sentences

def load_language_split(lang, split):
    filename = "dev.txt" if split == "validation" else f"{split}.txt"
    filepath = os.path.join(DATA_DIR, lang, filename)
    return read_conll(filepath)

# Collect all unique tags across all languages first (so labels are consistent)
all_tags = set()
raw_data = {"train": [], "validation": [], "test": []}

for lang in LANGUAGES:
    for split in ["train", "validation", "test"]:
        examples = load_language_split(lang, split)
        for ex in examples:
            all_tags.update(ex["ner_tags"])
        raw_data[split].extend(examples)
        print(f"Loaded {lang} {split}: {len(examples)} sentences")

label_list = sorted(all_tags)
print("\nCombined label set:", label_list)

features = Features({
    "tokens": Sequence(Value("string")),
    "ner_tags": Sequence(ClassLabel(names=label_list))
})

dataset_dict = DatasetDict({
    split: Dataset.from_list(raw_data[split], features=features)
    for split in ["train", "validation", "test"]
})

print(dataset_dict)
print("\nFirst training example:")
print(dataset_dict["train"][0])

print("\nNumber of sentences (combined):")
for split in ["train", "validation", "test"]:
    print(f"{split.capitalize()}: {len(dataset_dict[split])}")

# Save to disk so you can load it directly for fine-tuning
dataset_dict.save_to_disk("combined_ner_dataset")
print("\nSaved combined dataset to ./combined_ner_dataset")