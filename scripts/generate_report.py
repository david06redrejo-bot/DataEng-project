"""Generate the final HTML report with embedded figures, then convert to PDF."""
import base64, os

FIGS = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')
OUT = os.path.join(os.path.dirname(__file__), '..', 'reports')

def img_tag(name, caption=""):
    path = os.path.join(FIGS, name)
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'''<figure style="text-align:center;margin:20px 0;">
<img src="data:image/png;base64,{b64}" style="max-width:100%;border:1px solid #ddd;border-radius:4px;">
<figcaption style="font-style:italic;color:#555;margin-top:6px;">{caption}</figcaption>
</figure>'''

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Final Report — Spotify Music Genre Analysis</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
  body {{ font-family: 'Inter', sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px;
         color: #222; line-height: 1.7; font-size: 14px; }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 10px; font-size: 24px; }}
  h2 {{ color: #16213e; margin-top: 40px; border-bottom: 1px solid #ddd; padding-bottom: 6px; font-size: 18px; }}
  h3 {{ color: #0f3460; font-size: 15px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; }}
  th {{ background: #16213e; color: white; }}
  tr:nth-child(even) {{ background: #f9f9f9; }}
  .highlight {{ background: #e8f4f8; border-left: 4px solid #0f3460; padding: 12px 16px; margin: 16px 0; border-radius: 4px; }}
  @media print {{ body {{ margin: 20px; }} h1 {{ font-size: 20px; }} h2 {{ page-break-before: auto; }} }}
</style>
</head>
<body>

<h1>Final Report: Spotify Music Genre Analysis & Recommendation System</h1>
<p><strong>Authors:</strong> Pau Rossell & David Redrejo<br>
<strong>Course:</strong> Advanced Data Engineering<br>
<strong>Dataset:</strong> Spotify Tracks (~114,000 tracks × 20 variables)</p>

<hr>
<h2>1. Introduction</h2>
<p>This report presents the results of a complete data engineering pipeline applied to the Spotify Tracks dataset. 
The pipeline covers: Exploratory Data Analysis (EDA), Data Cleaning & Preprocessing, Dimensionality Reduction (PCA),
Supervised Classification (KNN), Unsupervised Clustering (K-Means), and a Content-Based Recommender System.
All methodologies follow the theoretical foundations documented in the Technical Design Report (context.md).</p>

<h2>2. Exploratory Data Analysis</h2>
<h3>2.1 Feature Distributions</h3>
<p>The 11 audio features show diverse distributions. Features like <code>danceability</code> and <code>valence</code>
are approximately uniform on [0,1], while <code>loudness</code> is left-skewed and <code>instrumentalness</code> 
is heavily zero-inflated.</p>
{img_tag('01_distributions.png', 'Figure 1: Audio feature distributions across ~114K tracks.')}

<h3>2.2 Correlation Structure</h3>
<p>The correlation matrix reveals two key relationships: <strong>energy↔loudness</strong> (r ≈ 0.76, strong positive)
and <strong>energy↔acousticness</strong> (r ≈ −0.73, strong negative). Most other features are weakly correlated,
confirming they capture independent acoustic dimensions.</p>
{img_tag('02_correlation.png', 'Figure 2: Pearson correlation matrix of audio features.')}

<h3>2.3 Genre Distribution</h3>
<p>The dataset contains <strong>114 genres</strong> with remarkably balanced distribution (~1,000 tracks per genre),
eliminating class-imbalance concerns for downstream classification.</p>
{img_tag('03_genre_distribution.png', 'Figure 3: Number of tracks per genre (balanced at ~1000).')}

<h3>2.4 Outlier Detection</h3>
<p>Box-plots identify outliers primarily in <code>speechiness</code>, <code>instrumentalness</code>, and
<code>liveness</code>. These are genuine data points (e.g., spoken-word tracks) rather than errors.</p>
{img_tag('04_boxplots.png', 'Figure 4: Box-plots for outlier detection across 8 features.')}

<h2>3. Data Cleaning & Preprocessing</h2>
<div class="highlight">
<strong>Pipeline:</strong> Missing values (MCAR → listwise deletion) → Duplicate removal → Median imputation → 
StandardScaler (z-score normalization) → LabelEncoder for genres.
</div>
<p>Missing values were identified exclusively in identity columns (<code>track_name</code>, <code>artists</code>) 
representing &lt;0.01% of data. Under the MCAR assumption, listwise deletion was applied. Feature scaling via 
StandardScaler is mandatory for PCA, KNN, and K-Means as all three rely on distance metrics.</p>

<h2>4. Dimensionality Reduction — PCA</h2>
<h3>4.1 Variance Analysis</h3>
<p>Principal Component Analysis on the 11 standardized audio features reveals that the first few components
capture the majority of variance. The cumulative explained variance plot identifies the optimal number of 
components to retain ≥90% of the information.</p>
{img_tag('06_pca_variance.png', 'Figure 5: Scree plot and cumulative explained variance.')}

<h3>4.2 2D Projection</h3>
<p>Projecting 5,000 sample tracks onto PC1-PC2 reveals partial genre separation — some genres (e.g., classical)
form distinct clusters, while others (rock, pop) overlap significantly.</p>
{img_tag('07_pca_2d.png', 'Figure 6: PCA 2D projection colored by genre.')}

<h3>4.3 Component Loadings</h3>
<p>The loadings heatmap shows which original features contribute most to each principal component. 
PC1 is dominated by the energy/loudness/acousticness axis, while PC2 captures rhythmic features.</p>
{img_tag('08_pca_loadings.png', 'Figure 7: PCA component loadings heatmap.')}

<h2>5. KNN Classification</h2>
<p>K-Nearest Neighbors was applied to a 5-genre subset (rock, pop, classical, hip-hop, jazz) with a 70/30
stratified train/test split.</p>

<h3>5.1 Hyperparameter Tuning</h3>
<p>5-fold stratified cross-validation was used to select the optimal k. The accuracy-vs-k curve identifies
the sweet spot balancing bias and variance.</p>
{img_tag('09_knn_tuning.png', 'Figure 8: KNN hyperparameter tuning — CV accuracy vs k.')}

<h3>5.2 Results</h3>
<p>The confusion matrix reveals that <strong>classical</strong> is the easiest genre to classify (acoustically 
distinct), while rock and pop are frequently confused (acoustic overlap).</p>
{img_tag('10_knn_confusion.png', 'Figure 9: KNN confusion matrix on test set.')}

<h2>6. K-Means Clustering</h2>
<h3>6.1 Optimal k Selection</h3>
<p>The Elbow Method (inertia/WCSS) and Silhouette Score analysis were used to determine the optimal number
of clusters.</p>
{img_tag('11_kmeans_elbow.png', 'Figure 10: Elbow method and silhouette analysis.')}

<h3>6.2 Cluster Visualization</h3>
<p>Comparing K-Means clusters (k=5) against true genre labels in PCA space shows that the algorithm
partially recovers the genre structure, though cluster boundaries differ from genre boundaries.</p>
{img_tag('12_kmeans_clusters.png', 'Figure 11: True genres vs K-Means clusters in PCA space.')}

<h3>6.3 Cluster-Genre Correspondence</h3>
{img_tag('13_cluster_genre.png', 'Figure 12: Cross-tabulation of clusters vs true genres.')}

<h2>7. Content-Based Recommender System</h2>
<p>A content-based recommendation engine was built using <strong>cosine similarity</strong> on the 11 standardized
audio features. Given a query track, the system returns the top-n most similar tracks.</p>

<h3>7.1 Genre Coherence Evaluation</h3>
<p>For 500 random query tracks, we measured the fraction of top-10 recommendations that share the query's genre. 
The distribution of this "genre coherence" metric validates the recommender's ability to surface acoustically
similar content.</p>
{img_tag('14_recommender_coherence.png', 'Figure 13: Recommender genre coherence distribution.')}

<h3>7.2 Similarity Heatmap</h3>
{img_tag('15_similarity_heatmap.png', 'Figure 14: Pairwise cosine similarity between 5 sample tracks.')}

<h2>8. Conclusions</h2>
<table>
<tr><th>Phase</th><th>Key Result</th></tr>
<tr><td>EDA</td><td>~114K tracks, 114 balanced genres. Strong energy↔loudness correlation.</td></tr>
<tr><td>PCA</td><td>11D compressed to fewer components retaining ≥90% variance.</td></tr>
<tr><td>KNN</td><td>Reasonable classification; classical easiest, rock/pop hardest to separate.</td></tr>
<tr><td>K-Means</td><td>Partial genre recovery; silhouette confirms moderate structure.</td></tr>
<tr><td>Recommender</td><td>Content-based cosine similarity achieves meaningful genre coherence.</td></tr>
<tr><td>Data Eng.</td><td>CSR sparse compression; artist collaboration graph modeled.</td></tr>
</table>

<p style="text-align:center;margin-top:40px;color:#888;font-size:12px;">
<em>Advanced Data Engineering — Pau Rossell & David Redrejo</em></p>

</body></html>"""

html_path = os.path.join(OUT, 'final_report.html')
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'HTML report written: {html_path}')
