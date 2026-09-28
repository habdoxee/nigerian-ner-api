import json, os, random, sys
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from pruned_embed import PrunedEmbedding, load_pruned_model

SRC = sys.argv[1] if len(sys.argv) > 1 else "ner_model_v3_caseaug"
OUT = "ner_model_pruned"
KEEP_TOP = 30000
DATA_DIRS = ["naija-ner", "correction_batch", "noisy_eval", "gazetteers"]
EXTS = (".json", ".jsonl", ".csv", ".tsv", ".txt", ".conll")
SKIP_NAMES = {"tokenizer.json", "tokenizer_config.json", "config.json",
              "special_tokens_map.json", "vocab.json", "added_tokens.json",
              "trainer_state.json", "vocab.txt", "merges.txt"}
SKIP_DIRS = (".git", "node_modules", "site-packages", "checkpoint-", "ner_model")


def walk_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_strings(v)


def collect_strings():
    found = set()
    for d in DATA_DIRS:
        for root, dirs, files in os.walk(d):
            dirs[:] = [x for x in dirs if not x.startswith(SKIP_DIRS)]
            for f in files:
                low = f.lower()
                if low in SKIP_NAMES or not low.endswith(EXTS):
                    continue
                p = os.path.join(root, f)
                if os.path.getsize(p) > 200 * 1024 * 1024:
                    continue
                text = open(p, encoding="utf-8", errors="ignore").read()
                if low.endswith(".json"):
                    try:
                        found.update(walk_strings(json.loads(text)))
                        continue
                    except ValueError:
                        pass
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if low.endswith(".jsonl"):
                        try:
                            found.update(walk_strings(json.loads(line)))
                            continue
                        except ValueError:
                            pass
                    found.add(line)
    return found


tok = AutoTokenizer.from_pretrained(SRC)
strings = list(collect_strings())
print(f"collected {len(strings)} unique strings")

variants = set()
for s in strings:
    s = s[:2000]
    variants.update((s, s.lower(), s.upper(), s.title()))
variants = list(variants)

keep = set(range(KEEP_TOP)) | set(tok.all_special_ids)
for i in range(0, len(variants), 2000):
    for ids in tok(variants[i:i + 2000], add_special_tokens=False)["input_ids"]:
        keep.update(ids)
keep = sorted(keep)
print(f"keeping {len(keep)} of the original vocabulary")

model = AutoModelForTokenClassification.from_pretrained(SRC).eval()

random.seed(0)
pool = [s for s in strings if 20 < len(s) < 300]
sample = random.sample(pool, min(300, len(pool)))


def preds(m):
    out = []
    with torch.inference_mode():
        for s in sample:
            enc = tok(s, return_tensors="pt", truncation=True, max_length=256)
            out.append(m(**enc).logits.argmax(-1))
    return out


before = preds(model)

old_w = model.get_input_embeddings().weight.data
old_vocab, hidden = old_w.shape
keep_t = torch.tensor(keep)
id_map = torch.full((old_vocab,), keep.index(tok.unk_token_id), dtype=torch.long)
id_map[keep_t] = torch.arange(len(keep))
emb = PrunedEmbedding(old_vocab, len(keep), hidden)
emb.id_map.copy_(id_map)
emb.weight.data.copy_(old_w[keep_t].half())
model.base_model.embeddings.word_embeddings = emb
model.config.vocab_size = len(keep)

os.makedirs(OUT, exist_ok=True)
torch.save(model.state_dict(), os.path.join(OUT, "pytorch_model.bin"))
model.config.save_pretrained(OUT)
tok.save_pretrained(OUT)

pruned = load_pruned_model(OUT)
after = preds(pruned)
agree = sum((a == b).sum().item() for a, b in zip(before, after))
total = sum(a.numel() for a in before)
size = os.path.getsize(os.path.join(OUT, "pytorch_model.bin")) / 1e6
print(f"saved {OUT}: weights {size:.0f} MB")
if total:
    print(f"token label agreement with original: {agree / total:.2%} on {len(sample)} sentences")
else:
    print("no sentence-length text found to compare; test by hand")
