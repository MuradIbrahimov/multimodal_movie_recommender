# Multimodal Movie Dataset v2

Built by `dataset/merge.ipynb`, the unified dataset creation notebook.

Construction summary:
- MovieLens provides collaborative ratings and stable movie IDs.
- MovieLens links normalize IMDb and TMDB IDs for cross-source joins.
- MovieGenreFromItsPoster contributes legacy poster URLs, IMDb scores, and poster genres.
- TheMoviesDataset metadata is used when available.
- TMDB_all_movies is applied in the same build as fallback metadata and as the preferred poster source.

Files:
- `final_movie_features_clean.csv`: one row per movie.
- `final_ratings_clean.csv`: all MovieLens ratings with user/movie indices.
- `model_sample_500k.csv`: sampled ratings joined with movie features.
- `movie2idx.csv`, `user2idx.csv`: index mappings.
- `audit_report.csv`: coverage and row-count checks.
- `poster_embeddings.npy`: created later by `extract_poster_embeddings.py`, not by this notebook.
- `model_df_clean.csv`: skipped by default because it is very large.

Next step:
- Run `python extract_poster_embeddings.py --data-dir dataset/MultimodalMovieDataset_v2` from the project root to add visual poster features.

Audit:
- movies: 62423
- ratings_rows: 25000095
- users: 162541
- poster_url_original: 36262
- tmdb_all_poster_url: 60960
- poster_url_final: 61824
- overview: 62034
- genres: 62136
- poster_source_tmdb_all: 60960
- poster_source_movies_metadata: 769
- poster_source_moviegenre_imdb: 95
- poster_source_missing: 599
- sample_rows: 500000
- sample_unique_movies: 18239
- sample_rows_with_poster_url: 499922
- sample_unique_movies_with_poster_url: 18208
