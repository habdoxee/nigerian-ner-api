"""
Extract a small, manageable batch from your large silver-labeled dataset
for MANUAL correction -- this is the real fix for code-mixing performance,
since self-training alone (all-silver, no human correction) cannot teach
the model patterns it didn't already know.

Pulls the first N examples per language from silver_training_data/*.conll
into a separate folder, ready for hand-correction. The remaining silver
examples are left as bulk (lower-quality but free) training volume.

Usage:
    python extract_correction_batch.py

Output:
    ./correction_batch/correction_hau.conll   (~100-150 examples)
    ./correction_batch/correction_ibo.conll
    ./correction_batch/correction_pcm.conll
    ./correction_batch/correction_yor.conll

After running: open each file in a text editor and correct the tags by
hand -- same process as your noisy_eval correction, just for TRAINING data
this time instead of evaluation data. Once corrected, run
combine_datasets_v2.py (not yet built -- ask for it once correction is done)
to merge: original clean data + hand-corrected batch (oversampled for more
influence) + remaining bulk silver data.
"""

import os

SILVER_DIR = "./silver_training_data"
OUTPUT_DIR = "./correction_batch"
EXAMPLES_PER_LANGUAGE = 120  # ~480 total across 4 languages -- a realistic
                             # manual correction workload, similar effort to
                             # your noisy_eval set

os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_conll_raw_blocks(filepath):
    """Returns the raw text blocks (one per sentence, including blank line
    separator) rather than parsed tokens -- preserves exact original
    formatting for easy re-insertion later."""
    blocks = []
    current_block_lines = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                if current_block_lines:
                    blocks.append("".join(current_block_lines))
                    current_block_lines = []
            else:
                current_block_lines.append(line)
        if current_block_lines:
            blocks.append("".join(current_block_lines))
    return blocks


for fname in sorted(os.listdir(SILVER_DIR)):
    if not fname.endswith(".conll"):
        continue
    lang_code = fname.replace("silver_", "").replace(".conll", "")
    filepath = os.path.join(SILVER_DIR, fname)

    blocks = read_conll_raw_blocks(filepath)
    batch = blocks[:EXAMPLES_PER_LANGUAGE]
    remaining = blocks[EXAMPLES_PER_LANGUAGE:]

    # Write the correction batch
    correction_path = os.path.join(OUTPUT_DIR, f"correction_{lang_code}.conll")
    with open(correction_path, "w", encoding="utf-8") as f:
        for block in batch:
            f.write(block)
            f.write("\n")
    print(f"{lang_code}: extracted {len(batch)} examples to {correction_path}")

    # Overwrite the original silver file with just the REMAINING (uncorrected)
    # examples, so combine_datasets_v2.py can treat correction_batch/ and
    # silver_training_data/ as two distinct, non-overlapping sources.
    with open(filepath, "w", encoding="utf-8") as f:
        for block in remaining:
            f.write(block)
            f.write("\n")
    print(f"  {fname} now holds the remaining {len(remaining)} uncorrected examples")

print(f"\nDone. Now manually correct the {EXAMPLES_PER_LANGUAGE * 4} examples")
print("across the 4 files in ./correction_batch/, then let me know so I can")
print("build the merge script that combines: clean data + corrected batch")
print("(oversampled) + remaining bulk silver data.")
