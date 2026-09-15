# auto_label_tweets.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import json

# 1. Load your fine-tuned model (update this path to your actual checkpoint dir)
MODEL_PATH = "./results/checkpoint-XXXX"  # e.g. wherever Trainer saved the best model

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

ner_pipeline = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",  # merges sub-word pieces into whole entities
)

# 2. Load Yoruba tweets from AfriSenti
tweets = load_dataset("HausaNLP/AfriSenti-Twitter", "yor", split="train")

print(f"Loaded {len(tweets)} Yoruba tweets")

# 3. Run NER on each tweet, convert to token/tag format matching your training schema
auto_labeled = []

for example in tweets:
    text = example["tweet"]  # check actual column name — AfriSenti sometimes uses "text"
    entities = ner_pipeline(text)

    # Build a simple record: raw text + extracted entities
    record = {
        "text": text,
        "entities": [
            {
                "entity_group": ent["entity_group"],  # e.g. PER, ORG, LOC, DATE
                "word": ent["word"],
                "start": ent["start"],
                "end": ent["end"],
                "score": float(ent["score"]),
            }
            for ent in entities
        ],
    }
    auto_labeled.append(record)

# 4. Save raw auto-labeled output for inspection
with open("yor_tweets_autolabeled.jsonl", "w", encoding="utf-8") as f:
    for rec in auto_labeled:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("Saved auto-labeled tweets to yor_tweets_autolabeled.jsonl")