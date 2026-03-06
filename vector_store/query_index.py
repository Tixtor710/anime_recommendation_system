import numpy as np
import faiss

embeddings = np.load("data/processed/anime_embeddings.npy")

index = faiss.read_index("data/processed/anime_index.faiss")


def similar_anime(anime_id, k=10):

    vector = embeddings[anime_id].reshape(1, -1)

    distances, indices = index.search(vector, k)

    return indices[0]


print(similar_anime(100))