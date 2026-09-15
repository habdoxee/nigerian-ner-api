"""
Auto-label Yoruba tweets from AfriSenti using a pretrained Yoruba NER model.

Produces a dataset in the same format your finetune_ner.py expects:
    {"tokens": [...], "ner_tags": [...]}  (ner_tags as label STRINGS here;
    convert to ids with your existing label2id mapping before training)

Run this with the SAME Python environment your other scripts use, e.g.:
    & "C:\\Users\\PC\\AppData\\Local\\Programs\\Python\\Python314\\python.exe" autolabel_yoruba_tweets.py
"""

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ---- Config ----
NER_MODEL_NAME = "mbeukman/xlm-roberta-base-finetuned-ner-yoruba"  # swap for your own checkpoint later
TWEET_DATASET = ("HausaNLP/AfriSenti-Twitter", "yor")
OUTPUT_DATASET_PATH = "yoruba_tweets_autolabeled"
MAX_EXAMPLES = None  # set an int (e.g. 2000) to limit for a quick test run

# ---- 1. Load tweets ----
print("Loading AfriSenti Yoruba tweets...")
tweet_ds = load_dataset(*TWEET_DATASET, split="train")
if MAX_EXAMPLES:
    tweet_ds = tweet_ds.select(range(min(MAX_EXAMPLES, len(tweet_ds))))
print(f"Loaded {len(tweet_ds)} tweets. Example: {tweet_ds[0]}")

# ---- 2. Load NER labeling model ----
print(f"Loading NER model: {NER_MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)

ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy=None,  # keep raw B-/I- tags, not merged spans
)

# ---- 3. Auto-label each tweet ----
def label_tweet(text):
    """
    Runs the NER pipeline on a raw tweet string and returns
    whitespace-split tokens aligned with predicted BIO tags,
    falling back to 'O' for any token the pipeline didn't tag.
    """
    if not text or not text.strip():
        return {"tokens": [], "ner_tags": []}

    predictions = ner_pipeline(text)
    tokens = text.split()

    # Build tags by matching predicted spans back onto whitespace tokens.
    # This is an approximation: subword models don't always align perfectly
    # with whitespace tokens, so some noise is expected and normal.
    tags = ["O"] * len(tokens)
    char_pos = 0
    token_spans = []
    for tok in tokens:
        start = text.find(tok, char_pos)
        end = start + len(tok)
        token_spans.append((start, end))
        char_pos = end

    for pred in predictions:
        p_start, p_end = pred["start"], pred["end"]
        label = pred["entity"]
        for i, (t_start, t_end) in enumerate(token_spans):
            if t_start < p_end and t_end > p_start:  # overlap
                tags[i] = label
                break

    return {"tokens": tokens, "ner_tags": tags}


print("Auto-labeling tweets... (this uses CPU/GPU depending on your torch install)")
results = {"tokens": [], "ner_tags": []}
text_column = "tweet" if "tweet" in tweet_ds.column_names else tweet_ds.column_names[0]

for i, example in enumerate(tweet_ds):
    labeled = label_tweet(example[text_column])
    if labeled["tokens"]:  # skip empty tweets
        results["tokens"].append(labeled["tokens"])
        results["ner_tags"].append(labeled["ner_tags"])
    if i % 500 == 0:
        print(f"  {i}/{len(tweet_ds)} processed")

# ---- 4. Save as a Hugging Face dataset on disk ----
auto_labeled_ds = Dataset.from_dict(results)
auto_labeled_ds.save_to_disk(OUTPUT_DATASET_PATH)
print(f"Saved {len(auto_labeled_ds)} auto-labeled examples to ./{OUTPUT_DATASET_PATH}")
print("Example:", auto_labeled_ds[0])
