# Project Implementation Plan: Spotify Music Recommender System

## 1. Goal

Build a **Content-Based Music Recommender System** using the Spotify Tracks Dataset. The system, given a song title, will return the top-N most similar songs by computing multi-dimensional similarity across structured audio features, sparse genre/artist encodings, and (optionally) cover art visual embeddings.

This project is designed to demonstrate all **four data archetypes** defined in `context.md`.

---

## 2. Architecture Overview

```
Raw CSV (114k tracks)
        │
        ▼
┌───────────────────────────┐
│  PHASE 1: STRUCTURED DATA │  Dense matrix of 11 audio features
│  (Archetype 1)            │  → Normalize with StandardScaler
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────┐
│  PHASE 2: SPARSE DATA     │  One-hot encode track_genre + artists
│  (Archetype 2)            │  → Store as CSR matrix (scipy.sparse)
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────┐
│  PHASE 3: UNSTRUCTURED    │  Fetch album cover art (M × N × 3)
│  (Archetype 3)            │  → Extract color histogram features
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────┐
│  PHASE 4: GRAPH MODELING  │  Build artist collaboration network
│  (Archetype 4)            │  → G = (V, E) using NetworkX
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────┐
│  FINAL: RECOMMENDER       │  scipy.sparse.hstack(all matrices)
│         ENGINE            │  → Cosine Similarity → Top-N Songs
└───────────────────────────┘
```

---

## 3. Detailed Phase Breakdown

### Phase 1: Structured Matrix (Archetype 1)
**File:** `notebooks/01_exploratory_data_analysis.ipynb`

| Step | Description |
|:--- |:--- |
| 1.1 | Load CSV into Pandas DataFrame (Observations × Variables) |
| 1.2 | Explore: `.shape`, `.describe()`, `.dtypes`, `.head()` |
| 1.3 | Identify missingness type (MCAR/MAR/MNAR) and remediate |
| 1.4 | Select the 11 numerical audio features |
| 1.5 | Scale features with `StandardScaler` (Z-score normalization) |

**Key audio features:** `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`

---

### Phase 2: Sparse Matrix Architectures (Archetype 2)
**File:** `notebooks/02_sparse_encoding.ipynb`

| Step | Description |
|:--- |:--- |
| 2.1 | One-hot encode `track_genre` (114 distinct genres) via `sklearn.OneHotEncoder(sparse_output=True)` |
| 2.2 | One-hot encode `artists` (multi-label, split on `;`) |
| 2.3 | Store both as **CSR matrices** (`scipy.sparse.csr_matrix`) |
| 2.4 | Inspect memory savings vs dense equivalent |
| 2.5 | Horizontally stack all feature matrices: `scipy.sparse.hstack([X_audio, X_genre, X_artists])` |
| 2.6 | Bridge to a SparsePandas DataFrame using `DataFrame.sparse.from_spmatrix` |

---

### Phase 3: Unstructured Media Processing (Archetype 3)
**File:** `notebooks/03_album_art_features.ipynb`

| Step | Description |
|:--- |:--- |
| 3.1 | Use Spotify Web API (or `spotipy`) to fetch album cover URLs |
| 3.2 | Download a sample subset of cover images as PNG (lossless) |
| 3.3 | Ingest as `M × N × 3` NumPy arrays |
| 3.4 | Extract color histograms (e.g., 16 bins per RGB channel = 48 features) |
| 3.5 | Concatenate visual features to the main recommendation matrix |

---

### Phase 4: Graph Modeling (Archetype 4)
**File:** `notebooks/04_artist_graph.ipynb`

| Step | Description |
|:--- |:--- |
| 4.1 | Parse `artists` column (split multi-artist tracks by `;`) |
| 4.2 | Build an **Undirected Weighted Graph** G=(V, E) using `networkx` |
| 4.3 | Nodes = distinct artists; Edges = collaborations; Weight = collaboration count |
| 4.4 | Compute centrality metrics (degree, betweenness) to identify hubs |
| 4.5 | Represent graph as an adjacency matrix for analysis |
| 4.6 | Visualize with `matplotlib` (top 30 most connected artists) |

---

### Phase 5: Recommender Engine
**File:** `notebooks/05_recommender_engine.ipynb`

| Step | Description |
|:--- |:--- |
| 5.1 | `hstack` all sparse feature matrices (audio + genre + artists + [optional: cover art]) |
| 5.2 | Compute the **Cosine Similarity** matrix using `sklearn.metrics.pairwise.cosine_similarity` |
| 5.3 | Implement `recommend(song_name, top_n=10)` function |
| 5.4 | Output: Top-N ranked tracks with similarity scores |

---

## 4. Technology Stack

| Library | Role |
|:--- |:--- |
| `pandas` | Core DataFrame operations and missingness handling |
| `numpy` | Low-level matrix arithmetic, image ingestion |
| `scipy.sparse` | CSR/CSC sparse storage and `hstack` |
| `scikit-learn` | Normalization, `OneHotEncoder`, `cosine_similarity` |
| `networkx` | Graph construction and centrality analysis |
| `matplotlib` | Visualization of EDA, histograms, graph topology |
| `Pillow` / `requests` | PNG image download and ingestion pipeline |

---

## 5. Missingness Strategy

| Column | Type | Strategy |
|:--- |:--- |:--- |
| `track_name`, `artists` | Identity | Listwise deletion (MCAR) |
| Numerical audio features | Sparse | Median imputation (MAR) |
| `popularity` | Numeric | Mean imputation (MCAR) |

---

## 6. Evaluation Criteria

- **Memory efficiency:** Confirm CSR matrix uses ≥80% less memory than dense equivalent
- **Relevance quality:** Manually validate top-10 recommendations for genre coherence
- **Pipeline integrity:** All 4 archetypes must be implemented and demonstrable
