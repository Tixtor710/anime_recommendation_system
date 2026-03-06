import pandas as pd
import numpy as np
import faiss
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# Load interactions
interactions = pd.read_csv("data/events/interactions.csv")

# Load mapping
user_map = pd.read_csv("data/processed/user_mapping.csv")
anime_map = pd.read_csv("data/processed/anime_mapping.csv")

user_to_index = dict(zip(user_map["user_index"], user_map["user_index"]))
anime_to_index = dict(zip(anime_map["anime_id"], anime_map["anime_index"]))

# Convert to matrix coordinates
rows = interactions["user_index"].map(user_to_index)
cols = interactions["anime_id"].map(anime_to_index)

matrix = csr_matrix(
    (np.ones(len(rows)), (rows, cols)),
    shape=(len(user_map), len(anime_map))
)

# Train collaborative model
svd = TruncatedSVD(n_components=64)
user_embeddings = svd.fit_transform(matrix)
anime_embeddings = svd.components_.T

# Save embeddings
np.save("data/processed/user_embeddings.npy", user_embeddings)
np.save("data/processed/anime_embeddings.npy", anime_embeddings)

# Build FAISS index
dimension = anime_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(anime_embeddings.astype("float32"))

faiss.write_index(index, "data/processed/anime_index.faiss")

print("Model retrained successfully.")