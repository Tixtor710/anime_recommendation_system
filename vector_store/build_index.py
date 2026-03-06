import numpy as np
import faiss

EMBEDDING_PATH = "data/processed/anime_embeddings.npy"


def load_embeddings():

    embeddings = np.load(EMBEDDING_PATH)

    print("Embeddings shape:", embeddings.shape)

    return embeddings
#============
#FAISS INDEX
#============
def build_index(embeddings):

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)

    index.add(embeddings)

    print("Total vectors indexed:", index.ntotal)

    return index
#======================
#save_index
#======================
def save_index(index):

    faiss.write_index(index, "data/processed/anime_index.faiss")

    print("Index saved.")

if __name__ == "__main__":

    embeddings = load_embeddings()

    index = build_index(embeddings)

    save_index(index)