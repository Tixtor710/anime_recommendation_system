import numpy as np

anime = np.load("data/processed/anime_embeddings.npy")
users = np.load("data/processed/user_embeddings.npy")

print("Anime embeddings:", anime.shape)
print("User embeddings:", users.shape)