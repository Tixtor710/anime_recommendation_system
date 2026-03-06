import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np

ANIMELIST_PATH = "data/raw/animelists_cleaned.csv"

def load_interactions():

    df = pd.read_csv(ANIMELIST_PATH)

    print("Raw interactions:", len(df))

    return df

def filter_interactions(df):

    df = df[df["my_score"] > 0]

    print("After removing unrated:", len(df))

    return df

def build_user_index(df):

    unique_users = df["username"].unique()

    user_to_index = {
        user: idx for idx, user in enumerate(unique_users)
    }

    df["user_index"] = df["username"].map(user_to_index)

    print("Total users:", len(user_to_index))

    return df, user_to_index

def build_anime_index(df):

    unique_anime = df["anime_id"].unique()

    anime_to_index = {
        anime: idx for idx, anime in enumerate(unique_anime)
    }

    df["anime_index"] = df["anime_id"].map(anime_to_index)

    print("Total anime:", len(anime_to_index))

    return df, anime_to_index

def build_sparse_matrix(df):

    rows = df["user_index"].values
    cols = df["anime_index"].values
    data = df["my_score"].values

    num_users = df["user_index"].max() + 1
    num_anime = df["anime_index"].max() + 1

    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(num_users, num_anime)
    )

    print("Matrix shape:", matrix.shape)

    return matrix

from scipy import sparse

def save_outputs(df, matrix, user_map, anime_map):

    df.to_csv(
        "data/processed/processed_interactions.csv",
        index=False
    )

    sparse.save_npz(
        "data/processed/user_item_matrix.npz",
        matrix
    )

    pd.DataFrame(
        user_map.items(),
        columns=["username", "user_index"]
    ).to_csv("data/processed/user_mapping.csv", index=False)

    pd.DataFrame(
        anime_map.items(),
        columns=["anime_id", "anime_index"]
    ).to_csv("data/processed/anime_mapping.csv", index=False)

if __name__ == "__main__":

    df = load_interactions()

    df = filter_interactions(df)

    df, user_map = build_user_index(df)

    df, anime_map = build_anime_index(df)

    matrix = build_sparse_matrix(df)

    save_outputs(df, matrix, user_map, anime_map)

    print("Pipeline complete.")