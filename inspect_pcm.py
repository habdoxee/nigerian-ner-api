import os

files = [
    "data/pcm/train.txt",
    "data/pcm/dev.txt",
    "data/pcm/test.txt"
]

for file in files:
    print("\n" + "=" * 60)
    print(file)
    print("=" * 60)

    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print("Number of lines:", len(lines))

    print("\nFirst 20 lines:")
    for line in lines[:20]:
        print(line.rstrip())