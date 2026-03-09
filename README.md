# Anime Recommendation System (FAISS + FastAPI)

A production-style anime recommendation system that combines **embedding-based collaborative filtering**, **FAISS vector search**, and a **FastAPI inference service**.

The system learns latent embeddings from user–anime interaction data and serves real-time recommendations through a lightweight REST API.

This project demonstrates an end-to-end ML engineering pipeline:

- Data ingestion
- Data preprocessing
- Embedding model training
- Vector indexing with FAISS
- Recommendation retrieval
- FastAPI deployment
- System benchmarking and evaluation


---

# System Architecture

![Architecture](docs/Architecture.png)

### Pipeline Overview

**1. Raw Data**

Input datasets include:

- Anime metadata
- User–anime interaction records

**2. Data Pipeline**

The preprocessing stage performs:

- Data cleaning
- Feature preparation
- User–item interaction matrix construction

**3. Embedding Model**

Matrix factorization generates latent embeddings for:

- Users
- Anime titles

These embeddings capture similarity relationships in the interaction space.

**4. Vector Store**

Anime embeddings are indexed using **FAISS** to enable fast similarity search.

This allows the system to retrieve nearest neighbors in milliseconds.

**5. API Layer**

A **FastAPI service** exposes the recommendation engine through REST endpoints.

The API supports:

- Similar anime search
- Personalized recommendations
- Interaction logging
- Trending detection


---

# Features

## Similar Anime Search

Retrieve anime similar to a given title using embedding similarity.

Endpoint:

```
GET /similar/{anime_id}
```

The system performs FAISS nearest-neighbor search on anime embeddings.

---

## Personalized Recommendations

Generate user-specific recommendations based on user embeddings.

Endpoint:

```
GET /recommend/{user_index}
```

The recommendation system:

- finds nearby anime embeddings
- filters previously watched titles
- removes invalid or missing entries

---

## Interaction Tracking

User activity is recorded for analytics and retraining.

Endpoint:

```
POST /interaction
```

Logged interactions include:

- watch
- click
- view

These records support:

- recommendation filtering
- trending analytics
- future model retraining

---

## Trending Anime Detection

Trending titles are computed based on recent user interactions.

Endpoint:

```
GET /trending
```

This feature helps highlight popular or rapidly growing anime titles.

---

# Performance

Vector search performance using **FAISS**.

| Metric | Value |
|------|------|
| Anime indexed | ~6,000 |
| Embedding dimension | 64 |
| Top-10 retrieval latency | ~1 ms |
| Index type | FAISS FlatL2 |

---

# Dataset Statistics

| Metric | Value |
|------|------|
| Anime titles | 6,143 |
| Users | 73,516 |
| Interaction records | ~7 million |
| Embedding dimension | 64 |

Dataset source:

Anime Recommendations Database  
https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database

Datasets are **not included in the repository** due to GitHub file size limits.

Place downloaded files in:

```
data/raw/
```

---

# Installation

### Install runtime dependencies

```
pip install -r requirements.txt
```

### Install development / training dependencies

```
pip install -r requirements-dev.txt
```

---

# Running the API

Start the FastAPI server:

```
uvicorn api.main:app --reload
```

Access interactive documentation:

```
http://127.0.0.1:8000/docs
```

---

# Example API Usage

Retrieve similar anime:

```
GET /similar/100
```

Example response:

```json
{
  "query_anime": 100,
  "recommendations": [
    {
      "anime_id": 104,
      "title": "Ayashi no Ceres",
      "score": 7.34
    }
  ]
}
```

---

# Live API Demo

Interactive API documentation:

https://web-production-b9e36.up.railway.app/docs

Example request:

```
curl https://web-production-b9e36.up.railway.app/recommend/10
```

Example response:

```json
{
  "user_index": 10,
  "recommendations": [
    {
      "anime_id": 5114,
      "title": "Fullmetal Alchemist: Brotherhood",
      "score": 9.2
    }
  ]
}
```

---

# Project Structure

```
anime_recommendation_system/
│
├── api/
│   └── main.py                 # FastAPI service
│
├── data_pipeline/
│   ├── ingest.py
│   ├── preprocess.py
│   ├── preprocess_interactions.py
│   └── build_matrix.py
│
├── models/
│   └── train_model.py          # embedding model training
│
├── vector_store/
│   ├── build_index.py
│   └── query_index.py
│
├── evaluation/
│   ├── benchmark.py
│   └── system_stats.py
│
├── docs/
│   └── architecture.png
│
└── README.md
```

---

# Evaluation

Benchmark scripts measure:

- vector search latency
- embedding statistics
- dataset coverage

Scripts available in:

```
evaluation/benchmark.py
evaluation/system_stats.py
```

---

# Future Improvements

Potential extensions for scaling and model quality:

- Hybrid recommendation system (content + collaborative filtering)
- Approximate FAISS indexes (IVF / HNSW)
- Online model retraining pipeline
- Real-time interaction streaming
- Dockerized deployment
