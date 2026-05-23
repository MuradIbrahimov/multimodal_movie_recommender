# Multimodal Movie Dataset v2

This dataset is rebuilt from the original `MultimodalMovieDataset` plus
`TMDB/TMDB_all_movies.csv` fallback metadata.

Important changes:
- `poster_url` now prefers TMDB image URLs from `TMDB_all_movies.csv`.
- The old IMDb/Amazon poster URL is preserved as `poster_url_original`.
- `poster_url_source` records where the selected poster URL came from.
- The original `user_idx` and `movie_idx` mappings are preserved.
- Full denormalized `model_df_clean.csv` is optional because it is very large.

Audit:
- movies: 59047
- poster_url_original: 35912
- tmdb_all_poster_url: 57721
- poster_url_final: 58558
- overview: 58677
- genres: 58766
- poster_source_tmdb_all: 57721
- poster_source_movies_metadata: 743
- poster_source_moviegenre_imdb: 94
- poster_source_missing: 489
- sample_rows: 500000
- sample_unique_movies: 18311
- sample_rows_with_poster_url: 499928
- sample_unique_movies_with_poster_url: 18280

Files:
- `final_movie_features_clean.csv`: one row per movie.
- `final_ratings_clean.csv`: copied rating interactions, if enabled.
- `model_sample_500k.csv`: 500k sampled ratings joined with v2 features.
- `movie2idx.csv`, `user2idx.csv`: unchanged mappings.
- `poster_embeddings.npy`: seeded from the previous embedding file if present.
- `model_df_clean.csv`: not written by default to avoid a huge file.

To fill the newly available TMDB poster rows, run from the project root:

```bash
python extract_poster_embeddings.py --data-dir dataset/MultimodalMovieDataset_v2
```

The extractor resumes from the seeded `poster_embeddings.npy` and only downloads
rows whose embeddings are still all zero. Use `--refresh-existing` only if you
want to recompute every poster embedding from the v2 URLs.
