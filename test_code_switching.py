"""
test_code_switching.py

Quick script to see how the NER model handles code-switched text
(sentences that mix Nigerian Pidgin, Yoruba, Hausa, Igbo, and/or English
within the same sentence).

It does two things:
1. Pulls a handful of real examples straight from your training dataset
   (combined_ner_dataset) so you can compare predictions vs. real labels.
2. Lets you drop in your own hand-written code-switched test sentences at
   the bottom, to see how the model reacts to mixes it may not have seen
   in training.
"""

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_DIR = "ner_model_pruned"   # change to "ner_model_final" to compare
DATASET_DIR = "combined_ner_dataset"
TOKEN_COLUMN = "tokens"
NUM_DATASET_EXAMPLES = 3

# common column names datasets use for NER labels -- auto-detected below,
# override here if none of these match yours
POSSIBLE_LABEL_COLUMNS = ["ner_tags", "labels", "tags", "ner_labels"]


print(f"Loading model + tokenizer from '{MODEL_DIR}' ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
model.eval()
id2label = model.config.id2label


def predict_and_print(words, gold_labels=None):
    """words: list[str] (already split into words), gold_labels: list[str] or None"""
    enc = tokenizer(words, is_split_into_words=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    pred_ids = logits.argmax(-1)[0].tolist()

    word_ids = enc.word_ids(batch_index=0)
    seen = set()
    print(f"{'TOKEN':<18}{'PRED':<12}{'GOLD':<12}")
    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen:
            continue
        seen.add(word_id)
        token_text = words[word_id]
        pred_label = id2label[pred_ids[idx]]
        gold_label = gold_labels[word_id] if gold_labels else ""
        print(f"{token_text:<18}{pred_label:<12}{gold_label:<12}")
    print()


# ---- 1. real examples straight from your dataset ----
print(f"Loading dataset from '{DATASET_DIR}' ...")
ds = load_from_disk(DATASET_DIR)
split = ds["train"] if hasattr(ds, "keys") else ds

label_column = next((c for c in POSSIBLE_LABEL_COLUMNS if c in split.column_names), None)
label_names = None
if label_column and hasattr(split.features[label_column], "feature"):
    label_names = split.features[label_column].feature.names

if label_column is None:
    print(f"  (no recognized label column found among {POSSIBLE_LABEL_COLUMNS} -- "
          f"showing predictions only, no gold labels. Columns available: {split.column_names})")

print(f"\n=== {NUM_DATASET_EXAMPLES} real dataset examples ===\n")
for i in range(min(NUM_DATASET_EXAMPLES, len(split))):
    example = split[i]
    words = example[TOKEN_COLUMN]
    gold_labels = None
    if label_column is not None:
        raw_labels = example[label_column]
        gold_labels = [label_names[l] for l in raw_labels] if label_names else raw_labels
    print(f"Sentence: {' '.join(words)}")
    predict_and_print(words, gold_labels)


# ---- 2. your own hand-written code-switched sentences ----
# Add real sentences you know mix languages, split into a list of words.
# Example structure (replace with real code-switched text you know):
#   ["Won", "lo", "school", "yesterday", "abeg"]

custom_sentences = [
    ["Won", "lo", "market", "kí", "wọ́n", "lè", "rí", "ẹrù"],
    ["Bola", "go", "to", "Lagos", "ni", "yesterday"],  # Example code-switched sentence
]

# Map your label IDs back to string labels
id2label = model.config.id2label

print("\n=== Custom Code-Switching Inferences ===")

# Set model to evaluation mode
model.eval()

for tokens in custom_sentences:
    # Tokenize input using word-level alignment
    inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    # Run forward pass through your model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted label IDs for each token position
    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
    
    # Handle single token edge case (where argmax returns a scalar)
    if isinstance(predictions, int):
        predictions = [predictions]

    # Align predictions back to original words (ignoring subword splits/special tokens)
    word_ids = inputs.word_ids(batch_index=0)
    previous_word_idx = None
    predicted_labels = []

    for idx, word_idx in enumerate(word_ids):
        # Skip special tokens (None) and handle only the first subtoken of each word
        if word_idx is not None and word_idx != previous_word_idx:
            pred_id = predictions[idx]
            label = id2label.get(pred_id, str(pred_id))
            predicted_labels.append(label)
            previous_word_idx = word_idx

    # Print results formatted like your evaluation script
    print(f"\nSentence: {' '.join(tokens)}")
    print(f"{'TOKEN':<20} {'PRED':<15}")
    print("-" * 35)
    for token, pred in zip(tokens, predicted_labels):
        print(f"{token:<20} {pred:<15}")

if custom_sentences:
    print("\n=== Custom code-switched test sentences ===\n")
    for words in custom_sentences:
        print(f"Sentence: {' '.join(words)}")
        predict_and_print(words)
else:
    print("\n(No custom sentences added yet -- edit `custom_sentences` in this "
          "script to test your own mixed-language examples.)")
