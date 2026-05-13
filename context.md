# Technical Design Report: Advanced Data Engineering for Music Classification

## Authors
Pau Rossell, David Redrejo & Joan Almirall

---

## 1. Project Overview and Strategic Objectives

This project implements a complete data engineering pipeline on the **Spotify Tracks Dataset** (~114,000 tracks × 20 variables). The work spans four phases: **Exploratory Data Analysis (EDA)**, **Data Cleaning & Preprocessing**, **Model Construction**, and **Model Validation**. Each phase is grounded in the theoretical foundations described below.

### Strategic Priorities

* **Exploratory Data Analysis:** Thorough statistical profiling and visualization of the feature space to uncover distributions, correlations, outliers, and class imbalances before any modeling is attempted.
* **Data Integrity:** Rigorous handling of missing values, duplicates, and type inconsistencies to produce a clean, analysis-ready dataset.
* **Dimensionality Reduction (PCA):** Application of Principal Component Analysis to compress the high-dimensional audio feature space while preserving the maximum variance, enabling both visualization and computational efficiency.
* **Classification (KNN):** Implementation of the K-Nearest Neighbors algorithm to classify tracks by genre, with proper hyperparameter tuning and evaluation.
* **Clustering (K-Means):** Unsupervised segmentation of the track space using K-Means clustering to discover latent groups in the data, evaluated against known genre labels.

---

## 2. Data Archetypes and Matrix Representation

The **Matrix of Observations** is the foundational unit of structured data engineering. This $\text{Rows} \times \text{Columns}$ model ($\text{Observations} \times \text{Variables}$) serves as the industry benchmark for relational analysis.

Within this framework, entities are rows and attributes are columns. For the Spotify dataset:

* **Variables:** `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `popularity`, `duration_ms`, `explicit`, `track_genre`.
* **Observations/Entities:** Each individual track.

From an architectural standpoint, dense matrices become memory-bound and computationally inefficient as dimensions expand, particularly when a large fraction of values are zero. Identifying these sparse regions is the prerequisite for transitioning to specialized compression algorithms.

---

## 3. Data Integrity: Handling Missingness and Imputation

Missing data introduces systematic bias if handled incorrectly. Proper categorization of the missingness mechanism determines the appropriate remediation strategy.

### Categorizing Missingness

* **MCAR (Missing Completely at Random):** Missingness is independent of any variable value, observed or hypothetical.
* **MAR (Missing at Random):** Propensity for missingness is related to observed data, but not the missing value itself.
* **MNAR (Missing Not at Random):** Systematic dependency where the missing value relates to the hypothetical value or other variables.

### Strategic Remediation

1. **Listwise Deletion (Complete-case analysis):** Acceptable only under MCAR. In other scenarios, it risks losing entire categories of observations.
2. **Pairwise Deletion:** Problematic in production; leads to inconsistencies where different model components use different observation counts.
3. **Imputation:** The preferred method for MAR and MNAR. Common strategies include:
   - *Mean/Median Imputation:* Replace missing values with the central tendency of the observed values in the same variable.
   - *KNN Imputation:* Use the K-nearest observations (by Euclidean or other distance) to infer missing values — leveraging the local structure of the data.

---

## 4. Feature Scaling and Normalization

Before applying distance-based algorithms (KNN, K-Means, PCA), all numerical features must be brought to a common scale. Without scaling, high-variance features (e.g., `tempo` ∼ 120) dominate low-variance features (e.g., `acousticness` ∼ 0.3).

### StandardScaler (Z-score normalization)

Transforms each feature $x$ to have zero mean and unit variance:

$$z = \frac{x - \mu}{\sigma}$$

This is the mandatory preprocessing step for PCA (which assumes centered data) and strongly recommended for KNN and K-Means.

### MinMaxScaler

Maps values to the $[0, 1]$ interval:

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

Useful when the algorithm requires bounded inputs but does not assume Gaussian distributions.

---

## 5. Principal Component Analysis (PCA)

### Theoretical Foundation

PCA is a linear dimensionality reduction technique that projects data onto a new set of orthogonal axes called **principal components**, ordered by the amount of variance they explain.

Given a centered data matrix $X \in \mathbb{R}^{n \times p}$, PCA computes the eigendecomposition of the covariance matrix:

$$C = \frac{1}{n-1} X^T X$$

The eigenvectors $v_1, v_2, \ldots, v_p$ of $C$ are the principal component directions, and the corresponding eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_p$ represent the variance captured by each component.

### Explained Variance Ratio

The fraction of total variance explained by the $k$-th component is:

$$\text{EVR}_k = \frac{\lambda_k}{\sum_{i=1}^{p} \lambda_i}$$

A common heuristic is to retain components until the **cumulative explained variance** exceeds a threshold (e.g., 90% or 95%).

### Strategic Application

* **Visualization:** Project high-dimensional data onto 2 or 3 principal components for scatter-plot visualization.
* **Noise Reduction:** Discard low-variance components that capture noise rather than signal.
* **Input Compression:** Feed a reduced feature matrix to downstream classifiers (KNN) or clustering algorithms (K-Means) to improve speed and mitigate the curse of dimensionality.

---

## 6. K-Nearest Neighbors (KNN)

### Theoretical Foundation

KNN is a **non-parametric, instance-based** (lazy) learning algorithm. It does not learn an explicit model; instead, it stores all training observations and classifies a new point $x$ by a majority vote among its $k$ nearest neighbors in the feature space.

### Distance Metrics

The most common distance metric is **Euclidean distance**:

$$d(x, y) = \sqrt{\sum_{i=1}^{p} (x_i - y_i)^2}$$

Other options include Manhattan distance ($L_1$) and Minkowski distance ($L_p$). Feature scaling is critical because unscaled features bias the distance computation.

### Hyperparameter: $k$

* **Small $k$** (e.g., 1–3): High variance, low bias — sensitive to noise and outliers.
* **Large $k$** (e.g., 20–50): Low variance, high bias — overly smooth decision boundaries.

The optimal $k$ is typically selected via **cross-validation** (e.g., 5-fold or 10-fold CV), plotting accuracy as a function of $k$ to find the elbow point.

### Evaluation Metrics

* **Accuracy:** Fraction of correct predictions. Misleading under class imbalance.
* **Precision, Recall, F1-score:** Per-class metrics that account for false positives and false negatives.
* **Confusion Matrix:** Full breakdown of predicted vs. actual labels.
* **Classification Report:** Aggregates precision, recall, and F1 across all classes (macro, weighted averages).

---

## 7. K-Means Clustering

### Theoretical Foundation

K-Means is an **unsupervised** partitioning algorithm that assigns $n$ observations to exactly $k$ clusters by minimizing the **within-cluster sum of squares (WCSS)**:

$$\text{WCSS} = \sum_{j=1}^{k} \sum_{x \in C_j} \| x - \mu_j \|^2$$

where $\mu_j$ is the centroid of cluster $C_j$.

### Lloyd's Algorithm

1. **Initialize** $k$ centroids (randomly, or via *k-means++* for smarter initialization).
2. **Assign** each observation to the nearest centroid.
3. **Update** each centroid to the mean of its assigned observations.
4. **Repeat** steps 2–3 until convergence (assignments no longer change) or a maximum iteration count is reached.

### Choosing $k$: The Elbow Method

Plot WCSS (inertia) as a function of $k$. The optimal $k$ is at the "elbow" — the point where increasing $k$ yields diminishing returns in variance reduction.

### Choosing $k$: The Silhouette Score

The **Silhouette Coefficient** for an observation $i$ is:

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$$

where $a(i)$ is the mean intra-cluster distance and $b(i)$ is the mean nearest-cluster distance. Values range from $-1$ (misclassified) to $+1$ (well-clustered). The mean silhouette score across all observations provides a global clustering quality metric.

### Evaluation Against Ground Truth

When true labels are available (as with `track_genre`), clustering quality can also be assessed via:

* **Adjusted Rand Index (ARI):** Measures agreement between cluster assignments and true labels, adjusted for chance.
* **Normalized Mutual Information (NMI):** Information-theoretic measure of label–cluster correspondence.

---

## 8. Miscellany

### Sparse Matrix Architectures

Sparse matrices — where the majority of elements are zero — require specialized storage to reduce memory footprint and enable faster computation. The principal formats are:

| Format | Full Name              | Strategic Application |
| :----- | :--------------------- | :-------------------- |
| **DOK** | Dictionary of Keys    | Incremental matrix construction. |
| **LIL** | List of Lists         | Row-wise modifications. |
| **COO** | Coordinate List       | Building matrices; requires sorting for efficient access. |
| **CSR** | Compressed Sparse Row | Numerical computation and row-wise operations. |
| **CSC** | Compressed Sparse Column | Column-wise operations and slicing. |

One-hot encoding categorical features (e.g., `track_genre`, `artists`) naturally produces very sparse matrices that benefit from CSR/CSC storage.

### Graph Modeling: Collaboration Networks

For datasets encoding 1:1 relations (e.g., artist collaborations), **Graphs** $G = (V, E)$ capture connections that flat matrices cannot efficiently model. Multi-artist tracks naturally define edges in a weighted collaboration graph, where the weight represents co-occurrence frequency. Key metrics include degree centrality, betweenness centrality, and connected-component analysis.

### Cross-Validation

To obtain unbiased estimates of model performance, we use **Stratified K-Fold Cross-Validation** (typically $k = 5$ or $k = 10$). Stratification ensures each fold preserves the class distribution of the full dataset, which is essential under class imbalance.

### The Curse of Dimensionality

As the number of features $p$ grows, the volume of the feature space increases exponentially, causing data points to become sparse. Distance-based methods (KNN, K-Means) degrade because distances become nearly uniform. PCA is one of the primary tools to combat this effect by projecting data onto the most informative subspace.

---

## 9. Technology Stack

* **Pandas:** High-level DataFrames for tabular data manipulation and analysis.
* **NumPy:** Low-level array operations and linear algebra.
* **SciPy:** Sparse matrix operations and scientific computing.
* **Scikit-learn:** Machine learning algorithms (PCA, KNN, K-Means), preprocessing (StandardScaler, LabelEncoder), model selection (cross_val_score, GridSearchCV), and evaluation metrics.
* **Matplotlib / Seaborn:** Statistical visualization.
* **NetworkX:** Graph construction and analysis.

---

## 10. Implementation Phases

1. **Phase 1 — Exploratory Data Analysis:** Statistical summaries, distribution plots, correlation analysis, outlier detection, class-balance assessment.
2. **Phase 2 — Data Cleaning & Preprocessing:** Missing-value remediation, duplicate removal, feature scaling (StandardScaler), label encoding.
3. **Phase 3 — Model Construction:**
   - PCA for dimensionality reduction and visualization.
   - KNN for genre classification with hyperparameter tuning.
   - K-Means for unsupervised clustering.
4. **Phase 4 — Model Validation:** Confusion matrices, classification reports, silhouette analysis, elbow plots, cross-validation scores.

### Evaluation Criteria

Success is measured by:
- **Classification accuracy and F1-score** of the KNN model.
- **Silhouette score and visual coherence** of K-Means clusters.
- **Variance preservation** in PCA dimensionality reduction.
- **Data integrity** ensured through rigorous missingness handling and preprocessing.
