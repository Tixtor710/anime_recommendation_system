import os
import requests

BASE_URL = "https://huggingface.co/joe77714/anime-recommender-artifacts/resolve/main/"

FILES = [
    "anime_embeddings.npy",
    "user_embeddings.npy",
    "anime_index.faiss",
    "anime_mapping.csv",
    "user_mapping.csv"
]

os.makedirs("data/processed", exist_ok=True)

for file in FILES:
    path = f"data/processed/{file}"

    if not os.path.exists(path):
        print(f"Downloading {file}")
        url = BASE_URL + file
        r = requests.get(url)

        with open(path, "wb") as f:
            f.write(r.content)

print("Model artifacts downloaded.")