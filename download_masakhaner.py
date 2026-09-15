from datasets import load_dataset

dataset = load_dataset("masakhane/masakhaner2", "pcm")

print(dataset)

print("\nFirst training example:")
print(dataset["train"][0])