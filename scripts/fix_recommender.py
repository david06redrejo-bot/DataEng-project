"""
Patch the recommender function in 01_exploratory_data_analysis.ipynb.

Changes:
  1. Averages feature vectors across ALL duplicate rows for the query song.
  2. Excludes every row sharing the exact (track_name, artists) pair from results.
  3. Deduplicates the result list by (track_name, artists) so no song appears twice.
  4. Replaces the single test cell with two cleaner example calls.
"""

import json
from pathlib import Path

NOTEBOOK = Path(r"C:\Users\david\Downloads\Labs 1 - Pandas (1)\PROJECT\notebooks\01_exploratory_data_analysis.ipynb")

NEW_RECOMMEND_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def recommend(song_name: str, df, X, top_n: int = 10):\n",
        "    \"\"\"\n",
        "    Content-based recommender using Cosine Similarity on the combined sparse feature matrix.\n",
        "\n",
        "    Improvements over naive version:\n",
        "    1. Averages the feature vector across ALL duplicate rows for the query song\n",
        "       (a song may appear many times in the dataset with slightly different metadata).\n",
        "    2. Excludes every row that shares the exact same (track_name, artists) pair as the\n",
        "       query, not just the first matching row.\n",
        "    3. Deduplicates the result list by (track_name, artists) so the same song is never\n",
        "       recommended twice, even when it exists in multiple genres/entries.\n",
        "    \"\"\"\n",
        "    import scipy.sparse as sp\n",
        "    from sklearn.metrics.pairwise import cosine_similarity\n",
        "\n",
        "    # --- 1. Locate all rows for the requested song ---\n",
        "    mask_query = df['track_name'].str.lower() == song_name.lower()\n",
        "    matches = df[mask_query]\n",
        "    if matches.empty:\n",
        "        print(f'Song \"{song_name}\" not found in dataset.')\n",
        "        return\n",
        "\n",
        "    # Use the (track_name, artists) of the highest-popularity entry as canonical identity\n",
        "    canonical = matches.loc[matches['popularity'].idxmax()]\n",
        "    query_track = canonical['track_name']\n",
        "    query_artist = canonical['artists']\n",
        "    print(f'Finding recommendations for: {query_track} \u2014 {query_artist}')\n",
        "\n",
        "    # --- 2. Build query vector as the mean of all matching rows ---\n",
        "    match_indices = matches.index.tolist()\n",
        "    song_vector = X[match_indices].mean(axis=0)          # dense (1 x n_features)\n",
        "    song_vector_sparse = sp.csr_matrix(song_vector)      # back to sparse for cosine_similarity\n",
        "\n",
        "    # --- 3. Compute cosine similarity against every track ---\n",
        "    scores = cosine_similarity(song_vector_sparse, X).flatten()\n",
        "\n",
        "    # --- 4. Mask out ALL rows that share (track_name, artists) with the query ---\n",
        "    exclude_mask = (\n",
        "        (df['track_name'].str.lower() == query_track.lower()) &\n",
        "        (df['artists'].str.lower() == query_artist.lower())\n",
        "    )\n",
        "    scores[exclude_mask] = -1  # force them below any real candidate\n",
        "\n",
        "    # --- 5. Rank all remaining tracks ---\n",
        "    ranked_indices = scores.argsort()[::-1]\n",
        "\n",
        "    results = df.loc[ranked_indices, ['track_name', 'artists', 'track_genre', 'popularity']].copy()\n",
        "    results['similarity'] = scores[ranked_indices].round(4)\n",
        "\n",
        "    # --- 6. Deduplicate by (track_name, artists) — keep highest similarity per unique song ---\n",
        "    results['_key'] = results['track_name'].str.lower() + '|||' + results['artists'].str.lower()\n",
        "    results = results.drop_duplicates(subset='_key', keep='first')\n",
        "    results = results.drop(columns='_key')\n",
        "\n",
        "    # Remove excluded rows (safety net) and return top-N\n",
        "    results = results[results['similarity'] >= 0].head(top_n)\n",
        "    return results.reset_index(drop=True)\n",
    ]
}

NEW_TEST_CELL_1 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# --- Test the recommender ---\n",
        "# Example 1: latin track that previously returned duplicates of itself\n",
        "recommend('La cancion', df, X_combined, top_n=10)\n",
    ]
}

NEW_TEST_CELL_2 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Example 2: classic rock track\n",
        "recommend('Bohemian Rhapsody', df, X_combined, top_n=10)\n",
    ]
}

nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
cells = nb["cells"]

# Find the recommend function cell and the test cell(s) that follow.
# We identify the recommend cell by looking for "def recommend" in the source.
recommend_idx = None
for i, cell in enumerate(cells):
    src = "".join(cell.get("source", []))
    if cell["cell_type"] == "code" and "def recommend" in src:
        recommend_idx = i
        break

if recommend_idx is None:
    raise RuntimeError("Could not find the 'def recommend' cell in the notebook.")

# Determine how many consecutive code cells after recommend_idx contain test calls
# (cells that reference 'recommend(' but not 'def recommend').
cells_to_remove_after = 0
for cell in cells[recommend_idx + 1:]:
    src = "".join(cell.get("source", []))
    if cell["cell_type"] == "code" and "recommend(" in src and "def recommend" not in src:
        cells_to_remove_after += 1
    else:
        break

print(f"Found 'def recommend' at cell index {recommend_idx}.")
print(f"Removing {cells_to_remove_after} test cell(s) after it and replacing with 2 new ones.")

# Splice: replace the old recommend cell + old test cells with the new ones
nb["cells"] = (
    cells[:recommend_idx]
    + [NEW_RECOMMEND_CELL, NEW_TEST_CELL_1, NEW_TEST_CELL_2]
    + cells[recommend_idx + 1 + cells_to_remove_after:]
)

NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook patched successfully.")
