import pandas as pd

ANIME_PATH = "data/raw/anime_cleaned.csv"
ANIMELIST_PATH = "data/raw/animelists_cleaned.csv"
USERS_PATH = "data/raw/users_cleaned.csv"


def load_data():
    anime = pd.read_csv(ANIME_PATH)
    animelists = pd.read_csv(ANIMELIST_PATH)
    users = pd.read_csv(USERS_PATH)

    return anime, animelists, users

def inspect_schema(anime, animelists, users):

    print("\nAnime columns:")
    print(anime.columns)

    print("\nAnimeList columns:")
    print(animelists.columns)

    print("\nUser columns:")
    print(users.columns)

def dataset_stats(anime, animelists, users):

    print("\nDataset Statistics")
    print("------------------")

    print("Total Anime:", anime["anime_id"].nunique())
    print("Total Users:", users["user_id"].nunique())
    print("Total Interactions:", len(animelists))

    print("\nSample interactions:")
    print(animelists.head())

if __name__ == "__main__":

    anime, animelists, users = load_data()

    inspect_schema(anime, animelists, users)

    dataset_stats(anime, animelists, users)