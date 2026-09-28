from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

tok = AutoTokenizer.from_pretrained("ner_model_pruned")
model = AutoModelForTokenClassification.from_pretrained("ner_model_pruned")
model.eval()

# use a real sentence from your training data, ideally one with known entities
text = "Wọlé lọ sí ilé-ìwé ní Èkó"  
inputs = tok(text, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
preds = logits.argmax(-1)[0]
tokens = tok.convert_ids_to_tokens(inputs["input_ids"][0])
for t, p in zip(tokens, preds):
    print(t, model.config.id2label[p.item()])