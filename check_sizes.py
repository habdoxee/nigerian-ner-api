from transformers import AutoModelForTokenClassification

SOURCE_MODEL_DIR = "ner_model_final"  # <-- set this to your actual model dir

model = AutoModelForTokenClassification.from_pretrained(SOURCE_MODEL_DIR)

total = 0
for name, p in model.named_parameters():
    size_mb = p.numel() * p.element_size() / 1024**2
    total += size_mb
    if "embedding" in name.lower():
        print(f"{name}: {size_mb:.1f} MB  (shape {tuple(p.shape)})")
print(f"\nTotal params: {total:.1f} MB")