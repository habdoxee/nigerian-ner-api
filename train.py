"""
Fine-tune a token-classification model on the combined MasakhaNER dataset
produced by prepare_dataset.py (./combined_ner_dataset).

COLAB VERSION — run this in a Google Colab notebook with a GPU runtime.

Before running, in a Colab cell:
    Runtime -> Change runtime type -> Hardware accelerator -> GPU (T4 is fine)

Then in separate Colab cells, BEFORE this script:

    # Cell 1: install deps
    !pip install -q transformers datasets seqeval accelerate evaluate torch

    # Cell 2: get your dataset into Colab -- pick ONE option:

    # Option A: upload a zipped version of combined_ner_dataset from your PC
    from google.colab import files
    uploaded = files.upload()   # select combined_ner_dataset.zip
    !unzip -q combined_ner_dataset.zip -d ./

    # Option B: mount Google Drive if the dataset folder already lives there
    from google.colab import drive
    drive.mount('/content/drive')
    # then set DATASET_DIR below to something like:
    # "/content/drive/MyDrive/babalola_codei/combined_ner_dataset"

Then run this script (paste into a cell, or %run train_colab.py if uploaded as a file).

Usage:
    python train_colab.py
"""

import os
import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

# ---------------------------------------------------------------------------
# Device check -- confirm Colab actually gave you a GPU
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    USE_FP16 = True
else:
    print("WARNING: No GPU detected. Go to Runtime -> Change runtime type -> GPU.")
    print("Training will fall back to CPU and be very slow.")
    USE_FP16 = False

# ---------------------------------------------------------------------------
# Config — adjust these if needed
# ---------------------------------------------------------------------------
MODEL_NAME = "Davlan/afro-xlmr-mini"   # smaller + tuned for African languages
DATASET_DIR = "./combined_ner_dataset"  # change if using Google Drive path
OUTPUT_DIR = "./results"
FINAL_MODEL_DIR = "/content/drive/MyDrive/ner_model_final"

NUM_EPOCHS = 10                 # bumped from 3 -- loss was still dropping sharply at epoch 3
BATCH_SIZE = 16                 # GPU can handle a bigger batch than CPU could
GRAD_ACCUM_STEPS = 1            # no longer needed for effective batch size w/ GPU
LEARNING_RATE = 3e-5            # bumped from 2e-5 -- head starts random, benefits from more signal early
MAX_LENGTH = 128

# This must match the label order printed by prepare_dataset.py:
# "Combined label set: [...]"
LABEL_LIST = [
    "B-DATE", "B-LOC", "B-ORG", "B-PER",
    "I-DATE", "I-LOC", "I-ORG", "I-PER",
    "O",
]
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
print(f"Loading dataset from {DATASET_DIR} ...")
dataset = load_from_disk(DATASET_DIR)
print(dataset)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=MAX_LENGTH,
        is_split_into_words=True,
    )

    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


print("Tokenizing and aligning labels...")
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_LIST),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
seqeval = evaluate.load("seqeval")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50,
    fp16=USE_FP16,             # mixed precision -- big speedup on Colab's GPU
    report_to="tensorboard",   # view loss/F1 curves with %load_ext tensorboard
    dataloader_num_workers=2,  # Colab gives you a couple of CPU cores to spare
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

print("\nEvaluating on test set...")
test_results = trainer.evaluate(tokenized_dataset["test"])
print(test_results)

# ---------------------------------------------------------------------------
# Save final model
# ---------------------------------------------------------------------------
trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print(f"\nTraining complete. Final model saved to: {FINAL_MODEL_DIR}")
print(f"Update auto_label_tweets.py -> MODEL_PATH = \"{FINAL_MODEL_DIR}\"")

# ---------------------------------------------------------------------------
# IMPORTANT: Colab's disk is wiped when the runtime disconnects.
# Copy your model out before you close the tab, e.g.:
#
#   from google.colab import drive
#   drive.mount('/content/drive')
#   !cp -r ./ner_model_final /content/drive/MyDrive/ner_model_final
#
# Or zip and download it directly:
#
#   !zip -r ner_model_final.zip ner_model_final
#   from google.colab import files
#   files.download('ner_model_final.zip')
# ---------------------------------------------------------------------------
