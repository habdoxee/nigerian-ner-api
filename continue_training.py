"""
Continue fine-tuning your EXISTING trained model on the combined
clean + silver-labeled (code-mixed) dataset, for domain adaptation to
noisy/code-mixed Nigerian tweets.

Key difference from your original train_colab.py: this loads
ner_model_final as the STARTING POINT (not the base afro-xlmr-mini), and
uses a LOWER learning rate with FEWER epochs -- standard practice for
continued fine-tuning, to adapt to the new domain without catastrophically
forgetting what the model already learned on clean text.

Run this in Colab, same as before: GPU runtime, dataset uploaded, etc.

Before running:
  1. Upload/unzip combined_ner_dataset_v2 (produced by combine_datasets.py)
     the same way you uploaded combined_ner_dataset before.
  2. Upload ner_model_final (or load it from your Google Drive if you
     saved it there) -- this script fine-tunes FROM this checkpoint.
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

if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    USE_FP16 = True
else:
    print("WARNING: No GPU detected.")
    USE_FP16 = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STARTING_CHECKPOINT = "./ner_model_final"       # your existing trained model
DATASET_DIR = "./combined_ner_dataset_v2"       # merged clean + silver data
OUTPUT_DIR = "./results_v2"
FINAL_MODEL_DIR = "/content/drive/MyDrive/ner_model_v2_codemixed"   # Colab: persists after runtime disconnects

# Continued fine-tuning uses a LOWER learning rate and FEWER epochs than
# training from scratch, to adapt gently rather than overwrite existing
# knowledge. If validation F1 on the ORIGINAL clean test set drops a lot
# after this, that's a sign of forgetting -- try fewer epochs or a lower LR.
NUM_EPOCHS = 4
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 1
LEARNING_RATE = 1e-5   # lower than the original 3e-5
MAX_LENGTH = 128

LABEL_LIST = [
    "B-DATE", "B-LOC", "B-ORG", "B-PER",
    "I-DATE", "I-LOC", "I-ORG", "I-PER",
    "O",
]
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}

print(f"Loading dataset from {DATASET_DIR} ...")
dataset = load_from_disk(DATASET_DIR)
print(dataset)

print(f"Loading tokenizer and model from checkpoint: {STARTING_CHECKPOINT} ...")
tokenizer = AutoTokenizer.from_pretrained(STARTING_CHECKPOINT)


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

model = AutoModelForTokenClassification.from_pretrained(
    STARTING_CHECKPOINT,   # continuing from your existing model, not the base one
    num_labels=len(LABEL_LIST),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
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
    fp16=USE_FP16,
    report_to="tensorboard",
    dataloader_num_workers=2,
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

print("Starting continued fine-tuning (code-mixing adaptation)...")
trainer.train()

print("\nEvaluating on ORIGINAL clean test set (compare to your 0.84 baseline)...")
test_results = trainer.evaluate(tokenized_dataset["test"])
print(test_results)
print("\nIMPORTANT: also re-run your noisy_eval (275/language hand-corrected)")
print("evaluation against this new model to see the noisy-text F1 improvement --")
print("that comparison is the actual evidence for your thesis's contribution.")

trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print(f"\nTraining complete. Final model saved to: {FINAL_MODEL_DIR}")
