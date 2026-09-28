"""
prune_vocab.py  (transformers 5.x compatible)

Shrinks an XLM-R-based token-classification model's vocabulary down to only
the subword pieces actually used by your training dataset (plus required
structural pieces: unknown / byte-fallback tokens), then rebuilds a matching
(much smaller) word-embedding table.

WHY THIS EXISTS
----------------
XLM-R ships with a 250,002-token multilingual vocabulary. For a model that
will only ever see Nigerian Pidgin, Yoruba, Hausa, and Igbo, well over 90%
of that vocabulary is dead weight -- it's what makes your "quantized" model
still ~400 MB (dynamic quantization only shrinks nn.Linear layers, it never
touches the embedding table).

HOW THIS VERSION WORKS (transformers 5.x)
-------------------------------------------
In transformers 5.x, XLMRobertaTokenizer is a thin wrapper around a Rust
`tokenizers` backend (accessible as `tokenizer._tokenizer`). Its Unigram
model stores the vocabulary as a plain JSON list of [piece, score] pairs,
where the LIST INDEX IS THE TOKEN ID directly -- no fairseq offset math
needed (that was only relevant for the old pure-Python/SentencePiece
implementation used in older transformers versions).

1. Loads your fine-tuned model + tokenizer from SOURCE_MODEL_DIR.
2. Tokenizes your ENTIRE training dataset (all splits) to find every token
   ID actually used.
3. Pulls the tokenizer's backend as JSON, filters its vocab list down to:
      - IDs seen in your dataset
      - all special token IDs (bos/pad/eos/unk/mask)
      - all byte-fallback pieces (pieces like "<0x0A>") so unseen
        characters at inference time still tokenize instead of erroring
4. Rebuilds the Rust tokenizer backend from the filtered JSON.
5. Builds a new (much smaller) embedding table by copying only the rows
   that correspond to kept token IDs from the original embedding table.
6. Saves the pruned tokenizer + model to OUTPUT_MODEL_DIR.

IMPORTANT -- READ BEFORE USING
--------------------------------
* This changes the model's input space. Any token NOT seen in your
  training dataset will now map to <unk> at inference time. Make sure your
  dataset scan covers realistic inputs -- if you have unlabeled text in
  the target languages beyond the NER-labeled set, add it to
  SUPPLEMENTARY_TEXT_FILES below.
* After pruning, run at least a short fine-tune (a few epochs on your
  existing training data) before deploying. The embedding rows are copied
  as-is, so the model should perform close to identically right away, but
  a short fine-tune settles anything that shifted at the margins.

USAGE
-----
    python prune_vocab.py
"""

import json
import os
import re

import torch
from datasets import load_from_disk
from tokenizers import Tokenizer
from transformers import AutoModelForTokenClassification, AutoTokenizer


# ----------------------------------------------------------------------
# CONFIG -- edit these to match your project
# ----------------------------------------------------------------------
SOURCE_MODEL_DIR = "ner_model_final"          # your fine-tuned model
DATASET_DIR = "combined_ner_dataset"          # HF dataset (load_from_disk)
TOKEN_COLUMN = "tokens"                       # column holding word lists
OUTPUT_MODEL_DIR = "ner_model_pruned"         # where the pruned model goes

# Optional: extra raw text files (one sentence/line each) in your target
# languages, beyond what's in the labeled dataset, to widen token coverage.
# Leave empty if you don't have any.
SUPPLEMENTARY_TEXT_FILES = [
    # "extra_yoruba_text.txt",
    # "extra_hausa_text.txt",
]

BYTE_FALLBACK_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def collect_used_token_ids(tokenizer, dataset_dir, token_column, extra_files):
    """Tokenize the whole dataset (+ any extra text files) and return the
    set of token IDs actually produced."""
    used_ids = set()

    print(f"Loading dataset from '{dataset_dir}' ...")
    ds = load_from_disk(dataset_dir)

    # ds may be a DatasetDict (train/validation/test) or a single Dataset
    splits = ds.values() if hasattr(ds, "values") else [ds]

    for split in splits:
        for example in split:
            words = example[token_column]
            enc = tokenizer(words, is_split_into_words=True, truncation=True)
            used_ids.update(enc["input_ids"])

    for path in extra_files:
        if not os.path.exists(path):
            print(f"  (skipping missing supplementary file: {path})")
            continue
        print(f"Scanning supplementary text file '{path}' ...")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                enc = tokenizer(line, truncation=True)
                used_ids.update(enc["input_ids"])

    print(f"Found {len(used_ids)} unique token IDs used in your data.")
    return used_ids


def build_pruned_tokenizer(tokenizer, used_ids):
    """Return (new backend Tokenizer object, old_id -> new_id map)."""
    data = json.loads(tokenizer._tokenizer.to_str())
    vocab = data["model"]["vocab"]  # list of [piece, score]; index == id
    unk_id = data["model"].get("unk_id")

    keep_ids = set(used_ids)
    keep_ids.update(tokenizer.all_special_ids)
    if unk_id is not None:
        keep_ids.add(unk_id)
    for idx, entry in enumerate(vocab):
        piece = entry[0]
        if BYTE_FALLBACK_RE.match(piece):
            keep_ids.add(idx)

    keep_ids = sorted(i for i in keep_ids if 0 <= i < len(vocab))
    pct = 100 * len(keep_ids) / len(vocab)
    print(f"Keeping {len(keep_ids)} of {len(vocab)} vocab entries ({pct:.1f}%).")

    old_to_new = {old: new for new, old in enumerate(keep_ids)}

    data["model"]["vocab"] = [vocab[old] for old in keep_ids]
    if unk_id is not None and unk_id in old_to_new:
        data["model"]["unk_id"] = old_to_new[unk_id]

    new_added_tokens = []
    for tok in data.get("added_tokens", []):
        old_id = tok["id"]
        if old_id in old_to_new:
            tok = dict(tok)
            tok["id"] = old_to_new[old_id]
            new_added_tokens.append(tok)
    data["added_tokens"] = new_added_tokens

    new_backend = Tokenizer.from_str(json.dumps(data))
    return new_backend, old_to_new


def main():
    print(f"Loading source model + tokenizer from '{SOURCE_MODEL_DIR}' ...")
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(SOURCE_MODEL_DIR)

    used_ids = collect_used_token_ids(
        tokenizer, DATASET_DIR, TOKEN_COLUMN, SUPPLEMENTARY_TEXT_FILES
    )

    new_backend, old_to_new = build_pruned_tokenizer(tokenizer, used_ids)
    new_vocab_size = new_backend.get_vocab_size()
    print(f"New tokenizer vocab size: {new_vocab_size} (was {tokenizer.vocab_size})")

    # mutate the tokenizer in place to use the pruned backend
    tokenizer._tokenizer = new_backend

    # ---- rebuild the embedding table ----
    old_embeddings = model.get_input_embeddings().weight.data
    hidden_dim = old_embeddings.shape[1]
    new_embeddings = torch.zeros((new_vocab_size, hidden_dim), dtype=old_embeddings.dtype)

    missing = 0
    for old_id, new_id in old_to_new.items():
        if old_id < old_embeddings.shape[0] and new_id < new_vocab_size:
            new_embeddings[new_id] = old_embeddings[old_id]
        else:
            missing += 1
    if missing:
        print(f"  (warning: {missing} id(s) could not be mapped -- check output)")

    model.resize_token_embeddings(new_vocab_size)
    model.get_input_embeddings().weight.data = new_embeddings
    model.config.vocab_size = new_vocab_size

    old_mb = old_embeddings.numel() * old_embeddings.element_size() / 1024**2
    new_mb = new_embeddings.numel() * new_embeddings.element_size() / 1024**2
    print(f"Embedding table: {old_mb:.1f} MB -> {new_mb:.1f} MB")

    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    print(f"Saving pruned model + tokenizer to '{OUTPUT_MODEL_DIR}' ...")
    model.save_pretrained(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)

    print("\nDone. Next steps:")
    print("  1. Sanity-check: tokenize a few known sentences with the new")
    print("     tokenizer and confirm entities still align as expected.")
    print("  2. Fine-tune for a few epochs on your training data to let the")
    print("     model settle into the new (smaller) embedding space.")
    print("  3. Re-run your quantization step on the fine-tuned pruned model")
    print("     -- quantize_dynamic() will now actually make a big dent,")
    print("     since the huge embedding table is gone.")


if __name__ == "__main__":
    main()
