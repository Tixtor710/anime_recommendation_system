import numpy as np
import faiss
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from datetime import datetime
import os

class InteractionEvent(BaseModel):
    user_index: int
    anime_id: int
    event: str
def get_user_history(user_index):

    path = "data/events/interactions.csv"

    try:
        df = pd.read_csv(
            path,
            names=["timestamp", "user_index", "anime_id", "event"]
        )
    except FileNotFoundError:
        return set()

    user_events = df[df["user_index"] == user_index]

    return set(user_events["anime_id"].values)
def get_popularity_scores():

    path = "data/events/interactions.csv"

    try:
        df = pd.read_csv(
            path,
            names=["timestamp", "user_index", "anime_id", "event"]
        )
    except FileNotFoundError:
        return {}

    popularity = df["anime_id"].value_counts()

    popularity_dict = popularity.to_dict()

    return popularity_dict
def get_trending_anime(limit=10):

    path = "data/events/interactions.csv"

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return []

    if df.empty:
        return []

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    cutoff = pd.Timestamp.now() - pd.Timedelta(hours=24)

    recent = df[df["timestamp"] >= cutoff]

    if recent.empty:
        return []

    popularity = recent["anime_id"].value_counts().head(limit)

    results = []

    for anime_id, count in popularity.items():

        if anime_id not in anime_df.index:
            continue

        row = anime_df.loc[anime_id]

        score = row["score"]
        score = None if pd.isna(score) else float(score)

        results.append({
            "anime_id": int(anime_id),
            "title": row["title"],
            "genre": row["genre"],
            "age_rating": row["rating"],
            "score": score,
            "watch_events": int(count)
        })

    return results
app = FastAPI()
#Load metadata
anime_df = pd.read_csv("data/raw/anime_cleaned.csv")

anime_df = anime_df.set_index("anime_id")

# Load embeddings
anime_embeddings = np.load("data/processed/anime_embeddings.npy")
user_embeddings = np.load("data/processed/user_embeddings.npy")



# Load vector index
index = faiss.read_index("data/processed/anime_index.faiss")
anime_map = pd.read_csv("data/processed/anime_mapping.csv")
index_to_anime = dict(
    zip(anime_map["anime_index"], anime_map["anime_id"])
)
user_map = pd.read_csv("data/processed/user_mapping.csv")

user_to_index = dict(zip(user_map["username"], user_map["user_index"]))
index_to_user = dict(zip(user_map["user_index"], user_map["username"]))
@app.get("/similar/{anime_id}")
def similar_anime(anime_id: int, k: int = 10):

    match = anime_map.loc[anime_map["anime_id"] == anime_id]

    if len(match) == 0:
        return {"error": "Anime ID not found"}

    anime_index = match["anime_index"].values[0]

    vector = anime_embeddings[anime_index].reshape(1, -1)

    distances, indices = index.search(vector, k+1)

    results = []

    for idx in indices[0]:

        real_id = index_to_anime.get(idx)

        if real_id is None:
           continue

        if real_id == anime_id:
            continue

        if real_id not in anime_df.index:
            continue

        row = anime_df.loc[real_id]
        score = row["score"]

        if pd.isna(score):
            score = None
        else:
            score = float(score)

        results.append({
    "anime_id": int(real_id),
    "title": row["title"],
    "genre": row["genre"],
    "age_rating": row["rating"],
    "score": score
})

    return {
        "query_anime": anime_id,
        "recommendations": results
    }
@app.get("/recommend/{user_index}")
def recommend(user_index: int, k: int = 10):

    # Validate user
    if user_index < 0 or user_index >= len(user_embeddings):
        return {"error": "User not found"}

    # Get user embedding
    user_vector = user_embeddings[user_index].reshape(1, -1)

    # Fetch watched history
    watched = get_user_history(user_index)

    # Popularity scores from interaction logs
    popularity = get_popularity_scores()

    # Retrieve candidate anime from FAISS
    distances, indices = index.search(user_vector, k * 5)

    candidates = []

    # Build candidate list with hybrid score
    for distance, idx in zip(distances[0], indices[0]):

        real_id = index_to_anime.get(idx)

        if real_id is None:
            continue

        if real_id in watched:
            continue

        if real_id not in anime_df.index:
            continue

        pop_score = popularity.get(real_id, 0)

        # Hybrid score
        hybrid_score = (0.7 * (1 - distance)) + (0.3 * pop_score)

        candidates.append((hybrid_score, real_id))

    # Rank candidates
    candidates.sort(reverse=True)

    results = []

    for _, real_id in candidates[:k]:

        row = anime_df.loc[real_id]

        score = row["score"]
        score = None if pd.isna(score) else float(score)

        results.append({
            "anime_id": int(real_id),
            "title": row["title"],
            "genre": row["genre"],
            "age_rating": row["rating"],
            "score": score
        })

    return {
        "user_index": user_index,
        "watched_count": len(watched),
        "recommendations": results
    }
@app.post("/interaction")
def log_interaction(event: InteractionEvent):

    if event.user_index < 0 or event.user_index >= len(user_embeddings):
        return {"error": "User not found"}

    if event.anime_id not in anime_df.index:
        return {"error": "Anime not found"}

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_index": event.user_index,
        "anime_id": event.anime_id,
        "event": event.event
    }

    path = "data/events/interactions.csv"

    os.makedirs("data/events", exist_ok=True)

    df = pd.DataFrame([record])

    file_exists = os.path.exists(path)

    df.to_csv(
    path,
    mode="a",
    header=not file_exists,
    index=False
)
    return {"status": "logged", "event": record}

@app.get("/trending")
def trending(limit: int = 10):

    results = get_trending_anime(limit)

    return {
        "time_window": "24_hours",
        "count": len(results),
        "trending": results
    }