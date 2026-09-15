from datasets import load_dataset

# Nigerian Pidgin
pidgin = load_dataset("masakhane/masakhaner", "pcm")

print(pidgin)

print("\nFirst training example:")
print(pidgin["train"][0])