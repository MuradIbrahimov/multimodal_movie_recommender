# Multimodal Movie Recommender

A neural collaborative filtering (NCF) system that fuses **user–item collaborative signals**, **movie metadata**, and **visual poster embeddings** to predict user ratings on the MovieLens 25M dataset.

---

## Overview

The project trains two models and compares their performance:

| Model | Description |
|---|---|
| **Baseline NCF** | Learns user and movie latent embeddings from rating history only |
| **Multimodal NCF** | Extends the baseline with a side branch that ingests genre vectors, TMDB metadata scalars, and MobileNetV2 poster embeddings |

The multimodal branch is trained as a residual correction on top of frozen collaborative-filtering weights, then fine-tuned end-to-end.

---

## Repository Structure

```
├── movie_recommender.ipynb          # Main training & evaluation notebook
├── extract_poster_embeddings.py     # Downloads posters & extracts MobileNetV2 embeddings
├── dataset/
│   ├── build_multimodal_dataset_v2.py   # Dataset construction script
│   ├── merge.ipynb                      # Merging MovieLens, TMDB & poster sources
│   └── MultimodalMovieDataset_v2/
│       └── README.md                    # Dataset-specific documentation
└── saved_models/
    ├── baseline_ncf.keras
    └── multimodal_ncf.keras
```

> **Note:** Raw datasets (ML25M, TMDB CSVs, poster images, processed `.csv`/`.npy` files) are excluded from version control via `.gitignore`. See *Dataset Setup* below.

---

## Model Architecture

### Baseline NCF
```
user_idx  ──► Embedding(32) ──┐
                               ├──► Dense(64) ──► Dense(32) ──► Dense(1) ──► rating
movie_idx ──► Embedding(32) ──┘
```

### Multimodal NCF (residual)
```
user_idx  ──► Embedding(32) ──┐
                               ├──► CF branch ──► cf_rating ──┐
movie_idx ──► Embedding(32) ──┘                               │
                                                               ├──► Add ──► rating
metadata  ──► BN ──► Dense(16) ──┐                            │
                                  ├──► Dense(32) ──► Dense(1) ──┘
poster    ──► BN ──► Dense(32) ──┘    (residual correction)
```

- **Poster embeddings**: MobileNetV2 (1280-d), optionally reduced to 64-d via TruncatedSVD
- **Metadata features**: one-hot genres + log-scaled & min-max-normalised `vote_average`, `vote_count`, `popularity`, `runtime`, `release_year`
- **Training strategy**: warm-start side branch with frozen CF weights → fine-tune full network

---

## Dataset Setup

### 1. MovieLens 25M
Download from [grouplens.org/datasets/movielens/25m](https://grouplens.org/datasets/movielens/25m/) and place files under `dataset/ML25M/`.

### 2. TMDB Metadata
Place `TMDB_all_movies.csv` under `dataset/TMDB/`.

### 3. MovieGenre Poster CSV
Place `MovieGenre.csv` under `dataset/MovieGenreFromItsPoster/`.

### 4. Build the multimodal dataset
```bash
cd dataset
python build_multimodal_dataset_v2.py
```

### 5. Extract poster embeddings
```bash
python extract_poster_embeddings.py --data-dir dataset/MultimodalMovieDataset_v2
```
The script downloads poster images concurrently and saves `poster_embeddings.npy` (shape `[N_movies, 1280]`). It resumes automatically on re-runs.

---

## Running the Notebook

```bash
# Install dependencies
pip install numpy pandas matplotlib scikit-learn tensorflow pillow requests

# Open notebook
jupyter lab movie_recommender.ipynb
```

Set `USE_SAMPLE = True` (default) to train on a 500 k-row sample. Set to `False` for the full dataset. The notebook auto-detects the data directory.

---

## Dependencies

| Package | Purpose |
|---|---|
| TensorFlow ≥ 2.x | Model training |
| NumPy / Pandas | Data handling |
| scikit-learn | Preprocessing, metrics, TruncatedSVD |
| Matplotlib | Visualisations |
| Pillow + Requests | Poster download & processing |

---

## Results

After training, both models are evaluated on a held-out test split (10%). Key metrics reported: **RMSE**, **MAE**, and training/validation loss curves.

Saved model weights are stored in `saved_models/` as `.keras` files.

---

## Authors

Group Project — Neural Networks (Master's Programme)
