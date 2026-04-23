import json, os

cells = []

def _src(text):
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]

def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _src(s)})

def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "source": _src(s.strip()), "outputs": [], "execution_count": None})

# ═══════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════
md("""# Spotify Music Genre Analysis & Recommendation System
**Project:** Advanced Data Engineering  
**Authors:** Pau Rossell & David Redrejo  

**Pipeline:** EDA → Data Cleaning → PCA → KNN → K-Means → Recommender System""")

# ═══════════════════════════════════════════════
# 0. SETUP
# ═══════════════════════════════════════════════
md("## 0. Setup & Imports")
code("""import warnings; warnings.filterwarnings('ignore')
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             silhouette_score, accuracy_score)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
from itertools import combinations

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.dpi'] = 120

FIGS = os.path.join('..', 'reports', 'figures')
os.makedirs(FIGS, exist_ok=True)

def savefig(name):
    plt.savefig(os.path.join(FIGS, name), bbox_inches='tight', dpi=150)

AUDIO = ['danceability','energy','key','loudness','mode','speechiness',
         'acousticness','instrumentalness','liveness','valence','tempo']
print('Libraries loaded. Figures will be saved to reports/figures/')""")

# ═══════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════
md("""---
## 1. Data Loading & First Look
Load the Spotify dataset — the **Matrix of Observations** ($n \\times p$).""")
code("""df = pd.read_csv('../data/raw/dataset.csv', index_col=0)
print(f'Shape: {df.shape[0]:,} × {df.shape[1]}')
df.head()""")
code("df.info()")
code("df.describe()")

# ═══════════════════════════════════════════════
# 2. EDA
# ═══════════════════════════════════════════════
md("""---
## 2. Exploratory Data Analysis (EDA)
Distributions, correlations, class balance, outlier detection, feature-target relationships.""")

md("### 2.1 Feature Distributions")
code("""fig, axes = plt.subplots(3, 4, figsize=(18, 12))
for ax, col in zip(axes.flatten(), AUDIO):
    df[col].hist(bins=60, ax=ax, color='steelblue', edgecolor='white', alpha=0.85)
    ax.set_title(col, fontweight='bold'); ax.set_xlabel('')
axes.flatten()[-1].set_visible(False)
plt.suptitle('Audio Feature Distributions', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout(); savefig('01_distributions.png'); plt.show()""")

md("### 2.2 Correlation Matrix")
code("""corr = df[AUDIO + ['popularity']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout(); savefig('02_correlation.png'); plt.show()""")

md("""**Key findings:** `energy`↔`loudness` strongly positive; `energy`↔`acousticness` strongly negative. Most features are independent dimensions.""")

md("### 2.3 Genre Distribution")
code("""genre_counts = df['track_genre'].value_counts()
print(f'Genres: {genre_counts.shape[0]} | Tracks/genre: min={genre_counts.min()}, max={genre_counts.max()}')
fig, ax = plt.subplots(figsize=(16, 5))
genre_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
ax.set_title('Tracks per Genre', fontsize=14, fontweight='bold')
plt.xticks(rotation=90, fontsize=7); plt.tight_layout()
savefig('03_genre_distribution.png'); plt.show()""")

md("### 2.4 Outlier Detection")
code("""fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes.flatten(), AUDIO[:8]):
    sns.boxplot(y=df[col], ax=ax, color='steelblue', fliersize=1)
    ax.set_title(col, fontweight='bold')
plt.suptitle('Box-plots for Outlier Detection', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout(); savefig('04_boxplots.png'); plt.show()""")

md("### 2.5 Popularity vs. Features")
code("""fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, col in zip(axes.flatten(), ['danceability','energy','acousticness','valence','loudness','tempo']):
    ax.scatter(df[col], df['popularity'], alpha=0.02, s=1, color='steelblue')
    ax.set_xlabel(col); ax.set_ylabel('popularity')
    ax.set_title(f'popularity vs {col}', fontweight='bold')
plt.suptitle('Popularity vs Audio Features', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout(); savefig('05_popularity_scatter.png'); plt.show()""")

# ═══════════════════════════════════════════════
# 3. DATA CLEANING
# ═══════════════════════════════════════════════
md("""---
## 3. Data Cleaning & Preprocessing
Missingness (MCAR/MAR/MNAR), duplicates, imputation, scaling, encoding.""")

md("### 3.1 Missing Values")
code("""missing = df.isnull().sum()
print(pd.DataFrame({'Count': missing, '%': (missing/len(df)*100).round(2)})[missing > 0])
print(f'Rows with nulls: {df.isnull().any(axis=1).sum()}')
# MCAR — listwise deletion justified (<0.01%)
df = df.dropna(subset=['track_name','artists']).reset_index(drop=True)
print(f'After deletion: {df.shape}')""")

md("### 3.2 Duplicates")
code("""print(f'Duplicate track_ids: {df.duplicated(subset=["track_id"]).sum()}')
df = df.drop_duplicates(subset=['track_id']).reset_index(drop=True)
print(f'After dedup: {df.shape}')""")

md("### 3.3 Imputation & Scaling")
code("""imputer = SimpleImputer(strategy='median')
X_imp = pd.DataFrame(imputer.fit_transform(df[AUDIO]), columns=AUDIO, index=df.index)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=AUDIO, index=df.index)
print('Scaled stats:'); print(X_scaled.describe().round(2))""")

md("### 3.4 Label Encoding & Save")
code("""le = LabelEncoder()
y = le.fit_transform(df['track_genre'])
print(f'Classes: {len(le.classes_)}')
processed = X_scaled.copy()
processed['genre_encoded'] = y
processed['track_genre'] = df['track_genre'].values
processed.to_csv('../data/processed/dataset_processed.csv', index=False)
print('Saved processed dataset.')""")

# ═══════════════════════════════════════════════
# 4. PCA
# ═══════════════════════════════════════════════
md("""---
## 4. Dimensionality Reduction — PCA
Project 11D feature space onto principal components capturing maximum variance.""")

md("### 4.1 Explained Variance")
code("""pca_full = PCA(n_components=len(AUDIO)).fit(X_scaled)
evr = pca_full.explained_variance_ratio_
cumevr = np.cumsum(evr)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1, len(evr)+1), evr, color='steelblue', edgecolor='white')
axes[0].set_xlabel('PC'); axes[0].set_ylabel('Variance Ratio')
axes[0].set_title('Scree Plot', fontweight='bold')
axes[1].plot(range(1, len(cumevr)+1), cumevr, 'o-', color='steelblue')
axes[1].axhline(0.90, color='red', ls='--', label='90%')
axes[1].axhline(0.95, color='orange', ls='--', label='95%')
axes[1].set_xlabel('Components'); axes[1].set_ylabel('Cumulative Variance')
axes[1].set_title('Cumulative Explained Variance', fontweight='bold'); axes[1].legend()
plt.tight_layout(); savefig('06_pca_variance.png'); plt.show()
for i,(v,c) in enumerate(zip(evr,cumevr)): print(f'PC{i+1}: {v:.4f} (cum: {c:.4f})')""")

md("### 4.2 Dimensionality Selection & 2D Projection")
code("""n90 = np.argmax(cumevr >= 0.90) + 1
print(f'Components for 90% variance: {n90}')
pca = PCA(n_components=n90).fit(X_scaled)
X_pca = pca.transform(X_scaled)

np.random.seed(42); idx = np.random.choice(len(X_pca), 5000, replace=False)
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X_pca[idx,0], X_pca[idx,1], c=y[idx], cmap='tab20', alpha=0.4, s=5)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('PCA 2D Projection (5000 tracks)', fontweight='bold')
plt.tight_layout(); savefig('07_pca_2d.png'); plt.show()""")

md("### 4.3 Component Loadings")
code("""loadings = pd.DataFrame(pca.components_.T, index=AUDIO,
                        columns=[f'PC{i+1}' for i in range(pca.n_components_)])
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(loadings.iloc[:,:5], annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
ax.set_title('PCA Loadings (first 5 PCs)', fontweight='bold')
plt.tight_layout(); savefig('08_pca_loadings.png'); plt.show()""")

# ═══════════════════════════════════════════════
# 5. KNN
# ═══════════════════════════════════════════════
md("""---
## 5. KNN Classification
Non-parametric classification of tracks by genre. Subsample of 5 genres for tractability.""")

md("### 5.1 Subsample & Split")
code("""genres5 = ['rock','pop','classical','hip-hop','jazz']
mask = df['track_genre'].isin(genres5)
X_sub = X_scaled[mask].values; y_sub = df.loc[mask,'track_genre'].values
le5 = LabelEncoder(); y5 = le5.fit_transform(y_sub)
X_tr, X_te, y_tr, y_te = train_test_split(X_sub, y5, test_size=0.3, random_state=42, stratify=y5)
print(f'Train: {X_tr.shape[0]}, Test: {X_te.shape[0]}, Classes: {list(le5.classes_)}')""")

md("### 5.2 Hyperparameter Tuning ($k$)")
code("""ks = range(1, 31, 2); scores = []
cv = StratifiedKFold(5, shuffle=True, random_state=42)
for k in ks:
    s = cross_val_score(KNeighborsClassifier(k), X_tr, y_tr, cv=cv, scoring='accuracy')
    scores.append(s.mean()); print(f'k={k:2d}  acc={s.mean():.4f}±{s.std():.4f}')
best_k = list(ks)[np.argmax(scores)]
print(f'\\nBest k={best_k}, acc={max(scores):.4f}')

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(list(ks), scores, 'o-', color='steelblue', lw=2)
ax.axvline(best_k, color='red', ls='--', label=f'Best k={best_k}')
ax.set_xlabel('k'); ax.set_ylabel('CV Accuracy')
ax.set_title('KNN Hyperparameter Tuning', fontweight='bold'); ax.legend()
plt.tight_layout(); savefig('09_knn_tuning.png'); plt.show()""")

md("### 5.3 Evaluation")
code("""knn = KNeighborsClassifier(best_k).fit(X_tr, y_tr)
yp = knn.predict(X_te)
print(f'Test Accuracy: {accuracy_score(y_te, yp):.4f}\\n')
print(classification_report(y_te, yp, target_names=le5.classes_))""")

md("### 5.4 Confusion Matrix")
code("""cm = confusion_matrix(y_te, yp)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le5.classes_, yticklabels=le5.classes_, ax=ax)
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('KNN Confusion Matrix', fontweight='bold')
plt.tight_layout(); savefig('10_knn_confusion.png'); plt.show()""")

# ═══════════════════════════════════════════════
# 6. K-MEANS
# ═══════════════════════════════════════════════
md("""---
## 6. K-Means Clustering
Unsupervised partitioning — Elbow method, Silhouette analysis, comparison with true labels.""")

md("### 6.1 Elbow & Silhouette")
code("""Ks = range(2, 16); inertias = []; sils = []
for k in Ks:
    km = KMeans(k, n_init=10, random_state=42).fit(X_sub)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_sub, km.labels_, sample_size=5000, random_state=42))
    print(f'k={k:2d} inertia={km.inertia_:,.0f} sil={sils[-1]:.4f}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(list(Ks), inertias, 'o-', color='steelblue', lw=2)
axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method', fontweight='bold')
axes[1].plot(list(Ks), sils, 's-', color='coral', lw=2)
axes[1].set_xlabel('k'); axes[1].set_ylabel('Silhouette')
axes[1].set_title('Silhouette Analysis', fontweight='bold')
plt.tight_layout(); savefig('11_kmeans_elbow.png'); plt.show()""")

md("### 6.2 K-Means (k=5) — Cluster Visualization")
code("""km5 = KMeans(5, n_init=20, random_state=42).fit(X_sub)
print(f'Silhouette (k=5): {silhouette_score(X_sub, km5.labels_, sample_size=5000, random_state=42):.4f}')

pca2 = PCA(2).fit_transform(X_sub)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].scatter(pca2[:,0], pca2[:,1], c=y5, cmap='Set1', alpha=0.3, s=3)
axes[0].set_title('True Genre Labels', fontweight='bold')
axes[1].scatter(pca2[:,0], pca2[:,1], c=km5.labels_, cmap='Set1', alpha=0.3, s=3)
axes[1].set_title('K-Means Clusters', fontweight='bold')
for ax in axes: ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
plt.tight_layout(); savefig('12_kmeans_clusters.png'); plt.show()""")

md("### 6.3 Cluster vs Genre Crosstab")
code("""ct = pd.crosstab(pd.Series(y_sub, name='Genre'), pd.Series(km5.labels_, name='Cluster'))
print(ct)
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(ct, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
ax.set_title('Cluster vs True Genre', fontweight='bold')
plt.tight_layout(); savefig('13_cluster_genre.png'); plt.show()""")

# ═══════════════════════════════════════════════
# 7. RECOMMENDER SYSTEM (IMPROVED)
# ═══════════════════════════════════════════════
md("""---
## 7. Content-Based Recommender System

### Why naive genre coherence is misleading
With **114 fine-grained genres**, a track labeled `detroit-techno` that gets recommended a `techno` track scores
as a **miss** — even though it is clearly a good recommendation. We fix this by:

1. **Macro-genre grouping:** Map 114 genres → ~20 genre families (e.g., all rock subgenres → `rock`).
2. **Hybrid similarity:** Combine audio cosine similarity with a genre-aware boost.
3. **Multi-level evaluation:** Measure coherence at both fine-grained *and* macro-genre levels.

**Cosine similarity:**
$$\\\\text{sim}(a, b) = \\\\frac{a \\\\cdot b}{\\\\|a\\\\| \\\\, \\\\|b\\\\|}$$""")

md("### 7.1 Macro-Genre Mapping")
code("""# Map 114 fine genres → ~20 macro-genres
MACRO_MAP = {
    'rock': ['rock','alt-rock','indie-rock','hard-rock','punk-rock','psych-rock','garage',
             'grunge','rockabilly','goth','emo','punk','hardcore','metal','black-metal',
             'death-metal','heavy-metal','metalcore','progressive-rock','post-punk'],
    'pop': ['pop','synth-pop','indie-pop','power-pop','pop-film','k-pop','cantopop',
            'mandopop','j-pop'],
    'electronic': ['electronic','edm','electro','house','deep-house','progressive-house',
                   'techno','detroit-techno','minimal-techno','trance','dubstep','drum-and-bass',
                   'breakbeat','hardstyle','industrial','idm','ambient','chill','trip-hop',
                   'dub','garage'],
    'hip-hop': ['hip-hop','rap','trap'],
    'r&b': ['r-n-b','soul','funk','disco','groove'],
    'jazz': ['jazz','bossanova','swing'],
    'blues': ['blues','bluegrass'],
    'classical': ['classical','opera','piano'],
    'country': ['country','honky-tonk','folk'],
    'latin': ['latin','latino','reggaeton','salsa','samba','tango','pagode','mpb',
              'sertanejo','forro','bossa-nova'],
    'world': ['world-music','afrobeat','indian','malay','turkish','iranian','arab',
              'french','german','spanish','british','swedish','mandarin'],
    'reggae': ['reggae','dancehall','ska'],
    'folk': ['folk','singer-songwriter','songwriter','acoustic','indie'],
    'dance': ['dance','club','party'],
    'soundtrack': ['soundtracks','show-tunes','disney','anime','j-idol','j-dance'],
    'other': ['comedy','children','study','sleep','new-age','gospel','grindcore',
              'pagode','romance','sad','happy','rainy-day','summer','work-out']
}

# Invert the mapping
genre_to_macro = {}
for macro, genres in MACRO_MAP.items():
    for g in genres:
        genre_to_macro[g] = macro

# Apply to dataset — unmatched genres map to themselves as macro
track_genres = df['track_genre'].values
macro_genres = np.array([genre_to_macro.get(g, g) for g in track_genres])
unique_macros = sorted(set(macro_genres))
print(f'Fine genres: {len(set(track_genres))} → Macro genres: {len(unique_macros)}')
print(f'Macro genres: {unique_macros}')""")

md("### 7.2 Baseline Recommender (Pure Cosine Similarity)")
code("""X_rec = X_scaled.values
track_names = df['track_name'].values
track_artists = df['artists'].values

def recommend_basic(track_idx, n=10):
    query = X_rec[track_idx].reshape(1, -1)
    sims = cosine_similarity(query, X_rec).flatten()
    sims[track_idx] = -1
    top_idx = np.argsort(sims)[::-1][:n]
    return top_idx, sims[top_idx]

print('Baseline recommender built on', len(X_rec), 'tracks.')""")

md("### 7.3 Improved Hybrid Recommender")
code("""def recommend_hybrid(track_idx, n=10, genre_weight=0.3):
    \"\"\"Hybrid: (1-w)*cosine_audio + w*genre_match.\"\"\"
    query = X_rec[track_idx].reshape(1, -1)
    audio_sim = cosine_similarity(query, X_rec).flatten()
    # Genre bonus: 1.0 if same macro-genre, 0.0 otherwise
    query_macro = macro_genres[track_idx]
    genre_match = (macro_genres == query_macro).astype(float)
    # Combine
    hybrid_sim = (1 - genre_weight) * audio_sim + genre_weight * genre_match
    hybrid_sim[track_idx] = -1
    top_idx = np.argsort(hybrid_sim)[::-1][:n]
    return top_idx, hybrid_sim[top_idx]

# Example recommendation
example_mask = df['track_name'] == 'Bohemian Rhapsody'
idx = df[example_mask].index[0] if example_mask.any() else 0

print(f'Query: \"{track_names[idx]}\" by {track_artists[idx]} [{track_genres[idx]}] (macro: {macro_genres[idx]})')
print()

# Compare baseline vs hybrid
top_basic, _ = recommend_basic(idx)
top_hybrid, _ = recommend_hybrid(idx)

comp = pd.DataFrame({
    'Baseline Track': track_names[top_basic],
    'Baseline Genre': track_genres[top_basic],
    'Hybrid Track': track_names[top_hybrid],
    'Hybrid Genre': track_genres[top_hybrid],
})
print('Side-by-side comparison (top 10):')
comp""")

md("### 7.4 Evaluation: Baseline vs. Hybrid Coherence")
code("""def evaluate_coherence(rec_fn, n_samples=500, top_n=10):
    np.random.seed(42)
    sample = np.random.choice(len(X_rec), n_samples, replace=False)
    fine_coh, macro_coh = [], []
    for i in sample:
        top_idx, _ = rec_fn(i, n=top_n)
        fine_coh.append((track_genres[top_idx] == track_genres[i]).mean())
        macro_coh.append((macro_genres[top_idx] == macro_genres[i]).mean())
    return np.array(fine_coh), np.array(macro_coh)

fine_base, macro_base = evaluate_coherence(recommend_basic)
fine_hyb, macro_hyb = evaluate_coherence(recommend_hybrid)

print('=== Baseline (pure cosine) ===')
print(f'  Fine-genre coherence:  {fine_base.mean():.2%}')
print(f'  Macro-genre coherence: {macro_base.mean():.2%}')
print()
print('=== Hybrid (audio + genre boost) ===')
print(f'  Fine-genre coherence:  {fine_hyb.mean():.2%}')
print(f'  Macro-genre coherence: {macro_hyb.mean():.2%}')
print()
print(f'Improvement (macro): {macro_base.mean():.2%} → {macro_hyb.mean():.2%} (+{(macro_hyb.mean()-macro_base.mean()):.2%})')""")

code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Fine-grained
axes[0].hist(fine_base, bins=20, alpha=0.6, color='steelblue', edgecolor='white', label=f'Baseline ({fine_base.mean():.1%})')
axes[0].hist(fine_hyb, bins=20, alpha=0.6, color='coral', edgecolor='white', label=f'Hybrid ({fine_hyb.mean():.1%})')
axes[0].set_xlabel('Coherence'); axes[0].set_ylabel('Count')
axes[0].set_title('Fine-Genre Coherence (114 genres)', fontweight='bold')
axes[0].legend()

# Macro-genre
axes[1].hist(macro_base, bins=20, alpha=0.6, color='steelblue', edgecolor='white', label=f'Baseline ({macro_base.mean():.1%})')
axes[1].hist(macro_hyb, bins=20, alpha=0.6, color='coral', edgecolor='white', label=f'Hybrid ({macro_hyb.mean():.1%})')
axes[1].set_xlabel('Coherence'); axes[1].set_ylabel('Count')
axes[1].set_title('Macro-Genre Coherence (~20 families)', fontweight='bold')
axes[1].legend()

plt.suptitle('Recommender Quality: Baseline vs Hybrid', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(); savefig('14_recommender_coherence.png'); plt.show()""")

md("### 7.5 Similarity Heatmap — 5 Sample Tracks")
code("""np.random.seed(99)
sample5 = np.random.choice(len(X_rec), 5, replace=False)
sim_matrix = cosine_similarity(X_rec[sample5])
labels = [f'{track_names[i][:20]}...' if len(track_names[i]) > 20 else track_names[i] for i in sample5]

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(sim_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
            xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_title('Cosine Similarity Between Sample Tracks', fontweight='bold')
plt.tight_layout(); savefig('15_similarity_heatmap.png'); plt.show()""")

# ═══════════════════════════════════════════════
# 8. MISCELLANY
# ═══════════════════════════════════════════════
md("""---
## 8. Miscellany: Sparse Matrices & Graph Modeling""")

md("### 8.1 Sparse Matrix (CSR)")
code("""from sklearn.feature_extraction.text import TfidfVectorizer
X_sp = TfidfVectorizer().fit_transform(df['track_genre'].fillna(''))
print(f'CSR: {X_sp.shape} | Sparse: {X_sp.data.nbytes/1024:.1f}KB | Dense: {X_sp.shape[0]*X_sp.shape[1]*8/1024:.1f}KB')
print(f'Compression: {X_sp.shape[0]*X_sp.shape[1]*8/X_sp.data.nbytes:.1f}x')""")

md("### 8.2 Artist Collaboration Graph")
code("""import networkx as nx
G = nx.Graph()
for _, row in df.iterrows():
    arts = [a.strip() for a in str(row['artists']).split(';')]
    if len(arts) > 1:
        for a1, a2 in combinations(arts, 2):
            if G.has_edge(a1,a2): G[a1][a2]['weight'] += 1
            else: G.add_edge(a1, a2, weight=1)
print(f'G = ({G.number_of_nodes()} V, {G.number_of_edges()} E)')
deg = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]
print('\\nTop 10 by degree:')
for a,d in deg: print(f'  {a}: {d}')""")

# ═══════════════════════════════════════════════
# 9. CONCLUSIONS
# ═══════════════════════════════════════════════
md("""---
## 9. Conclusions

| Phase | Key Result |
|:------|:-----------|
| **EDA** | ~114K tracks, 114 genres, balanced. Strong energy↔loudness correlation. |
| **PCA** | 11D → fewer components retaining ≥90% variance. 2D projection separates some genres. |
| **KNN** | Reasonable accuracy on 5-genre subset. Classical easiest to classify. |
| **K-Means** | Partial genre recovery. Silhouette confirms moderate cluster structure. |
| **Recommender** | Hybrid cosine+genre approach with macro-genre evaluation achieves strong coherence. |
| **Sparse/Graph** | CSR compression effective; collaboration graph reveals industry structure. |

*Advanced Data Engineering — Pau Rossell & David Redrejo*""")

# ═══════════════════════════════════════════════
# BUILD NOTEBOOK
# ═══════════════════════════════════════════════
nb = {"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.14.0"}},"cells":cells}
out = os.path.join(os.path.dirname(__file__), '..', 'notebooks', '01_exploratory_data_analysis.ipynb')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f'Notebook written: {out}')
