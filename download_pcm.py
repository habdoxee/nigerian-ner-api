import requests
import os

base_url = "https://raw.githubusercontent.com/masakhane-io/masakhane-ner/main/MasakhaNER2.0/data/pcm"

output_dir = "data/pcm"
os.makedirs(output_dir, exist_ok=True)

files = {
    "train.txt": f"{base_url}/train.txt",
    "dev.txt": f"{base_url}/dev.txt",
    "test.txt": f"{base_url}/test.txt",
}

for filename, url in files.items():
    print(f"Downloading {filename}...")

    response = requests.get(url)

    if response.status_code == 200:
        path = os.path.join(output_dir, filename)

        with open(path, "wb") as f:
            f.write(response.content)

        print(f"Saved: {path}")
    else:
        print(f"Failed: HTTP {response.status_code}")

print("\nDownload complete.")