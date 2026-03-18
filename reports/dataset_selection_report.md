# Dataset Selection Report: Spotify Tracks Dataset

## 1. Executive Summary

Following our technical design requirements to process and synthesize heterogeneous data formats, we have selected the **Spotify Tracks Dataset** for our Advanced Data Engineering project. This dataset contains over 114,000 distinct tracks, encompassing a rich variety of numerical audio features, categorical metadata, and implicit relational structures.

The strategic reason for choosing this dataset is that it natively contains, or allows for the extraction of, all four primary data archetypes outlined in our Technical Design Report (Structured Matrices, Sparse Architectures, Unstructured Media, and Relational Graphs). This enables a comprehensive end-to-end pipeline without needing to merge completely unrelated datasets.

## 2. Strategic Mapping to Data Archetypes

We will process this dataset to demonstrate proficiency in the four required formats:

### 2.1. Standard Matrix Observations (Structured Data)
The foundational representation of this data will be a dense matrix of observations where:
*   **Entities (Rows):** Individual songs (e.g., *track_id*).
*   **Attributes (Columns):** The core audio features extracted by Spotify's algorithms (`danceability`, `energy`, `valence`, `tempo`, `acousticness`, etc.).
**Engineering Plan:** We will perform Exploratory Data Analysis (EDA) on this dense matrix to identify correlations between audio features and track popularity. We will apply rigorous statistical methods to identify and impute any missingness (MCAR, MAR, MNAR) if present.

### 2.2. Sparse Matrix Architectures
The dataset contains highly categorical and high-dimensional features, specifically the `track_genre` and the `artists` columns. If we were to one-hot encode every artist or genre across 114,000 tracks, the resulting dense matrix would be computationally inefficient and memory-bound.
**Engineering Plan:** We will convert these categorical associations into **Compressed Sparse Row (CSR)** formats using `scipy.sparse`. This will drastically reduce memory footprint when preparing the dataset for machine learning models (like genre classification or popularity prediction), demonstrating $O(n)$ efficiency.

### 2.3. Unstructured Data Synthesis (Media Integration)
While the CSV itself is structured, the entities represent rich media.
**Engineering Plan:** To fulfill the unstructured media requirement, we will utilize the `track_id` and `album_name` metadata to synthesize unstructured inputs. We can do this by executing a script to dynamically download a subset of **Album Cover Art (Images)**. These raw pixel grids ($M \times N \times 3$) will then undergo feature extraction (e.g., color histograms or deep-learned visual descriptors) to see if album visual aesthetics correlate with the track's audio `valence` or `energy`.

### 2.4. Network and Relational Modeling (Graphs)
The `artists` column frequently contains multiple collaborators separated by semicolons (e.g., `Brandi Carlile;Sam Smith`). This represents a 1:1 relational network of musicians.
**Engineering Plan:** We will extract these entities to build an **Undirected Graph** $G = (V, E)$ using NetworkX:
*   **Vertices ($V$):** Distinct Artists.
*   **Edges ($E$):** A connection indicating an edge when two artists collaborate on a track.
*   **Weight:** The number of times they have collaborated or the average popularity of their joint tracks.
We can then transition this into an Adjacency Matrix to identify central nodes (the most collaborative artists) within the industry graph.

## 3. Ingestion Infrastructure

To ensure a reproducible, robust, and scalable data pipeline, we have implemented an ingestion script derived from [tttza/kaggle_dataset_downloader](https://github.com/tttza/kaggle_dataset_downloader/blob/main/download_dataset.py). 
This script performs a three-stage ingestion process:
1. **API Interfacing:** Dynamically interprets whether the target is a dataset or a specific competition.
2. **Staged Downloading:** Fetches the `.zip` archive into a secure, temporary `./download` cache.
3. **Automated Extraction:** Programmatically unzips the files and routes them into the `../data/raw/[dataset-name]` directory using Python's native `zipfile` and `shutil` libraries.

This optimization ensures our pipeline operates smoothly without relying on system-level `unzip` commands, maximizing cross-platform compatibility across Windows and Linux environments.

## 4. Conclusion

The Spotify Tracks dataset perfectly aligns with our infrastructure goals. It allows us to move from standard Pandas DataFrames into Scipy sparse arrays, ingest unstructured media via album art, and leverage graph theory to map the music industry—all while maintaining high code performance and analytical integrity.
