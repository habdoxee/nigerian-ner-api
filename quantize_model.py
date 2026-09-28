"""
Quantize the fine-tuned NER model to shrink its size/memory footprint so it
fits within Render's free-tier 512MB RAM limit.

Run this ONCE after training, before deploying. It reads from ner_model_final
and writes a new, smaller model to ner_model_quantized.

Usage:
    python quantize_model.py

Requires:
    pip install torch transformers
"""

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

SOURCE_MODEL_DIR = "./ner_model_final"
QUANTIZED_MODEL_DIR = "./ner_model_quantized"

print(f"Loading model from {SOURCE_MODEL_DIR} ...")
model = AutoModelForTokenClassification.from_pretrained(SOURCE_MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL_DIR)

print("Applying dynamic quantization (float32 -> int8 for Linear layers)...")
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # quantize the weight-heavy Linear layers
    dtype=torch.qint8,
)

print(f"Saving quantized model to {QUANTIZED_MODEL_DIR} ...")
quantized_model.config.save_pretrained(QUANTIZED_MODEL_DIR)
tokenizer.save_pretrained(QUANTIZED_MODEL_DIR)
torch.save(quantized_model.state_dict(), f"{QUANTIZED_MODEL_DIR}/pytorch_model.bin")

print("\nDone. Compare file sizes:")
import os
def folder_size_mb(path):
    total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs)
    return total / (1024 * 1024)

print(f"  Original:  {folder_size_mb(SOURCE_MODEL_DIR):.1f} MB")
print(f"  Quantized: {folder_size_mb(QUANTIZED_MODEL_DIR):.1f} MB")