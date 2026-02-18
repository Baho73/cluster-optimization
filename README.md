# Cluster Optimization

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn) ![License: MIT](https://img.shields.io/github/license/Baho73/cluster-optimization) for Text Embeddings ![CI](https://github.com/Baho73/cluster-optimization/actions/workflows/ci.yml/badge.svg)

End-to-end clustering pipeline for 45K text embeddings: data cleaning with ensemble outlier detection, optimal cluster count selection using 4 methods, and final KMeans clustering with t-SNE visualization.

## Pipeline

```
Raw Data (45,895 texts with embeddings)
    │
    ▼
┌─────────────────────────────────┐
│  1. Data Cleaning               │
│  ├─ KNN Distance Outliers       │
│  ├─ Local Outlier Factor (LOF)  │
│  └─ Isolation Forest            │
│  → Ensemble vote (≥2 of 3)     │
└─────────────────────────────────┘
    │  43,304 clean samples (-5.6%)
    ▼
┌─────────────────────────────────┐
│  2. Optimal k Selection         │
│  ├─ Elbow Method                │
│  ├─ Silhouette Analysis         │
│  ├─ Calinski-Harabasz Index     │
│  └─ Davies-Bouldin Index        │
│  → Median of 4 methods          │
└─────────────────────────────────┘
    │  k = 57
    ▼
┌─────────────────────────────────┐
│  3. Final Clustering            │
│  ├─ KMeans (k=57, k-means++)   │
│  ├─ t-SNE 2D visualization     │
│  └─ Cluster analysis & export  │
└─────────────────────────────────┘
    │  57 clusters with centroids
    ▼
  Output CSV
```

## Data Cleaning

Three outlier detection methods are applied independently, and a **conservative ensemble voting** approach removes points flagged by at least 2 out of 3 methods:

| Method | Outliers Detected |
|--------|----------------:|
| KNN Distance (z-score > 1.0) | 4,729 |
| Local Outlier Factor | 4,590 |
| Isolation Forest | 4,590 |
| **Combined (≥2 votes)** | **2,591** |

Result: 45,895 → **43,304 clean samples** (5.6% removed)

## Optimal k Selection

Four standard methods evaluate cluster counts from 2 to 80:

| Method | Optimal k |
|--------|----------:|
| Elbow Method | 2 |
| Silhouette Analysis | 2 |
| Calinski-Harabasz Index | 57 |
| Davies-Bouldin Index | 57 |
| **Final (median)** | **57** |

## Results

- **57 clusters** from 43,304 text embeddings (3,072-dimensional)
- Cluster sizes range from ~100 to ~1,600 items
- t-SNE visualization confirms well-separated cluster structure

## Tech Stack

- **Python** &mdash; core language
- **scikit-learn** &mdash; KMeans, t-SNE, LOF, Isolation Forest, silhouette/CH/DB metrics
- **pandas** / **NumPy** &mdash; data manipulation
- **matplotlib** / **seaborn** &mdash; visualization
- **SciPy** &mdash; statistical aggregation (mode)

## Project Structure

```
├── cluster_optimization.ipynb   # Full pipeline notebook
└── README.md
```

## Usage

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy
```

Open `cluster_optimization.ipynb` in Jupyter or Google Colab and run all cells. The notebook expects a CSV file with columns: `_id`, `text`, `n_tokens`, `embedding` (string representation of float arrays).
