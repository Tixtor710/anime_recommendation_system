import numpy as np
import time
import faiss

anime_embeddings = np.load("data/processed/anime_embeddings.npy")

index = faiss.read_index("data/processed/anime_index.faiss")

query = anime_embeddings[100].reshape(1, -1)

start = time.time()

for _ in range(1000):
    index.search(query, 10)

end = time.time()

avg_latency = (end - start) / 1000

print("Average search latency:", avg_latency)