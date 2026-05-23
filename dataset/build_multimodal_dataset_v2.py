"""
Build an enriched v2 multimodal movie dataset.

Why this exists:
The original merge notebook loaded TMDB_all_movies.csv but did not use it as a
fallback source. That left many movies without usable poster URLs and metadata,
even though TMDB_all_movies can fill most of them.

This script starts from the existing MultimodalMovieDataset outputs, enriches
the movie-level table with TMDB_all_movies, and writes a new dataset folder:

    dataset/MultimodalMovieDataset_v2/

The v2 dataset keeps the original MovieLens user/movie index mappings so it
stays compatible with existing rating files and recommender code.
"""

from __future__ import annotations

import argparse
import ast
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"
DEFAULT_SAMPLE_SIZE = 500_000
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build MultimodalMovieDataset_v2.")
    parser.add_argument("--dataset-dir", type=Path, default=here)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=here / "MultimodalMovieDataset",
        help="Existing clean dataset folder.",
    )
    parser.add_argument(
        "--tmdb-all",
        type=Path,
        default=here / "TMDB" / "TMDB_all_movies.csv",
        help="Large TMDB metadata CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "MultimodalMovieDataset_v2",
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument(
        "--write-full-model-df",
        action="store_true",
        help="Also write the full denormalized model_df_clean.csv. This is huge.",
    )
    parser.add_argument(
        "--no-copy-ratings",
        action="store_true",
        help="Do not copy final_ratings_clean.csv into the v2 folder.",
    )
    return parser.parse_args()


def clean_string(value: object) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none", "<na>"} else np.nan


def normalize_tmdb_series(series: pd.Series) -> pd.Series:
    out = (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    return out.mask(out.isin(["", "nan", "None", "<NA>"]))


def normalize_imdb_series(series: pd.Series) -> pd.Series:
    out = (
        series.astype("string")
        .str.strip()
        .str.replace("tt", "", regex=False)
        .str.replace(r"\.0$", "", regex=True)
        .str.lstrip("0")
    )
    return out.mask(out.isin(["", "nan", "None", "<NA>"]))


def parse_listish(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None
    if isinstance(parsed, list):
        return [str(v).strip() for v in parsed if str(v).strip()]
    return [part.strip() for part in text.split("|") if part.strip()]


def parse_comma_genres(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def standardize_genres(genres: list[str]) -> list[str]:
    genre_map = {
        "Musical": "Music",
        "Sci-Fi": "Science Fiction",
        "Children": "Family",
        "Film-Noir": "Noir",
    }
    out: list[str] = []
    for genre in genres:
        clean = genre_map.get(str(genre).strip(), str(genre).strip())
        if clean and clean not in out:
            out.append(clean)
    return out


def combine_genres(row: pd.Series) -> list[str]:
    combined: list[str] = []
    for column in (
        "meta_genres_list",
        "tmdb_all_genres_list",
        "poster_genres_list",
        "ml_genres_list",
    ):
        for genre in row.get(column, []):
            if genre not in combined:
                combined.append(genre)
    return combined


def poster_url_from_path(path: object) -> object:
    path = clean_string(path)
    if pd.isna(path):
        return np.nan
    text = str(path)
    if text.startswith("http://") or text.startswith("https://"):
        return text
    if not text.startswith("/"):
        text = "/" + text
    return f"{TMDB_IMAGE_BASE}{text}"


def first_non_null(*values: object) -> object:
    for value in values:
        cleaned = clean_string(value)
        if pd.notna(cleaned):
            return cleaned
    return np.nan


def read_tmdb_all(tmdb_all_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = [
        "id",
        "title",
        "vote_average",
        "vote_count",
        "release_date",
        "runtime",
        "imdb_id",
        "original_language",
        "original_title",
        "overview",
        "popularity",
        "genres",
        "poster_path",
    ]
    tmdb = pd.read_csv(tmdb_all_path, usecols=usecols, low_memory=False)
    tmdb["tmdb_id_clean"] = normalize_tmdb_series(tmdb["id"])
    tmdb["imdb_id_clean"] = normalize_imdb_series(tmdb["imdb_id"])

    tmdb["tmdb_score"] = (
        tmdb["poster_path"].notna().astype(int) * 16
        + tmdb["overview"].notna().astype(int) * 8
        + tmdb["genres"].notna().astype(int) * 4
        + tmdb["vote_count"].notna().astype(int) * 2
        + tmdb["title"].notna().astype(int)
    )
    tmdb = tmdb.sort_values("tmdb_score", ascending=False)

    renamed = tmdb.rename(
        columns={
            "title": "tmdb_all_title",
            "original_title": "tmdb_all_original_title",
            "genres": "tmdb_all_genres",
            "poster_path": "tmdb_all_poster_path",
            "vote_average": "tmdb_all_vote_average",
            "vote_count": "tmdb_all_vote_count",
            "popularity": "tmdb_all_popularity",
            "runtime": "tmdb_all_runtime",
            "release_date": "tmdb_all_release_date",
            "overview": "tmdb_all_overview",
            "original_language": "tmdb_all_original_language",
        }
    )
    keep = [
        "tmdb_id_clean",
        "imdb_id_clean",
        "tmdb_all_title",
        "tmdb_all_original_title",
        "tmdb_all_genres",
        "tmdb_all_poster_path",
        "tmdb_all_vote_average",
        "tmdb_all_vote_count",
        "tmdb_all_popularity",
        "tmdb_all_runtime",
        "tmdb_all_release_date",
        "tmdb_all_overview",
        "tmdb_all_original_language",
    ]
    by_tmdb = renamed[keep].dropna(subset=["tmdb_id_clean"]).drop_duplicates(
        "tmdb_id_clean"
    )
    by_imdb = renamed[keep].dropna(subset=["imdb_id_clean"]).drop_duplicates(
        "imdb_id_clean"
    )
    return by_tmdb, by_imdb


def enrich_movie_features(source_dir: Path, tmdb_all_path: Path) -> tuple[pd.DataFrame, dict]:
    movies = pd.read_csv(source_dir / "final_movie_features_clean.csv", low_memory=False)
    movies["tmdb_key"] = normalize_tmdb_series(movies["tmdb_id_clean"])
    movies["imdb_key"] = normalize_imdb_series(movies["imdb_id_clean"])
    movies["poster_url_original"] = movies["poster_url"]

    tmdb_by_tmdb, tmdb_by_imdb = read_tmdb_all(tmdb_all_path)
    movies = movies.merge(
        tmdb_by_tmdb,
        left_on="tmdb_key",
        right_on="tmdb_id_clean",
        how="left",
        suffixes=("", "_tmdb_join"),
    )
    movies = movies.merge(
        tmdb_by_imdb.add_suffix("_imdb_join"),
        left_on="imdb_key",
        right_on="imdb_id_clean_imdb_join",
        how="left",
    )

    def fill_from_tmdb(column: str, tmdb_column: str, imdb_column: str) -> None:
        movies[column] = movies[column].where(movies[column].notna(), movies[tmdb_column])
        movies[column] = movies[column].where(movies[column].notna(), movies[imdb_column])

    fill_from_tmdb("final_title", "tmdb_all_title", "tmdb_all_title_imdb_join")
    fill_from_tmdb(
        "original_title",
        "tmdb_all_original_title",
        "tmdb_all_original_title_imdb_join",
    )
    fill_from_tmdb("meta_overview", "tmdb_all_overview", "tmdb_all_overview_imdb_join")
    fill_from_tmdb(
        "meta_original_language",
        "tmdb_all_original_language",
        "tmdb_all_original_language_imdb_join",
    )

    numeric_pairs = {
        "meta_vote_average": ("tmdb_all_vote_average", "tmdb_all_vote_average_imdb_join"),
        "meta_vote_count": ("tmdb_all_vote_count", "tmdb_all_vote_count_imdb_join"),
        "meta_popularity": ("tmdb_all_popularity", "tmdb_all_popularity_imdb_join"),
        "meta_runtime": ("tmdb_all_runtime", "tmdb_all_runtime_imdb_join"),
    }
    for target, (tmdb_column, imdb_column) in numeric_pairs.items():
        movies[target] = pd.to_numeric(movies[target], errors="coerce")
        movies[tmdb_column] = pd.to_numeric(movies[tmdb_column], errors="coerce")
        movies[imdb_column] = pd.to_numeric(movies[imdb_column], errors="coerce")
        fill_from_tmdb(target, tmdb_column, imdb_column)

    release_year = pd.to_numeric(movies["release_year"], errors="coerce")
    tmdb_year = pd.to_datetime(movies["tmdb_all_release_date"], errors="coerce").dt.year
    imdb_year = pd.to_datetime(
        movies["tmdb_all_release_date_imdb_join"], errors="coerce"
    ).dt.year
    movies["release_year"] = release_year.where(release_year.notna(), tmdb_year)
    movies["release_year"] = movies["release_year"].where(
        movies["release_year"].notna(), imdb_year
    )

    # Prefer TMDB poster URLs because the older Amazon poster URLs often fail.
    tmdb_poster_path = movies["tmdb_all_poster_path"].where(
        movies["tmdb_all_poster_path"].notna(), movies["tmdb_all_poster_path_imdb_join"]
    )
    movies["tmdb_all_poster_path_final"] = tmdb_poster_path
    movies["tmdb_all_poster_url"] = tmdb_poster_path.apply(poster_url_from_path)

    current_meta_url = movies["meta_poster_path"].apply(poster_url_from_path)
    original_url = movies["poster_url_original"].apply(clean_string)
    movies["poster_url"] = movies["tmdb_all_poster_url"].where(
        movies["tmdb_all_poster_url"].notna(), current_meta_url
    )
    movies["poster_url"] = movies["poster_url"].where(
        movies["poster_url"].notna(), original_url
    )
    movies["poster_url_source"] = np.select(
        [
            movies["tmdb_all_poster_url"].notna(),
            current_meta_url.notna(),
            original_url.notna(),
        ],
        ["tmdb_all", "movies_metadata", "moviegenre_imdb"],
        default="missing",
    )

    movies["meta_poster_path"] = movies["meta_poster_path"].where(
        movies["meta_poster_path"].notna(), tmdb_poster_path
    )

    for column in ("ml_genres_list", "poster_genres_list", "meta_genres_list"):
        movies[column] = movies[column].apply(parse_listish).apply(standardize_genres)
    tmdb_genres = movies["tmdb_all_genres"].where(
        movies["tmdb_all_genres"].notna(), movies["tmdb_all_genres_imdb_join"]
    )
    movies["tmdb_all_genres_list"] = tmdb_genres.apply(parse_comma_genres).apply(
        standardize_genres
    )
    movies["final_genres_list"] = movies.apply(combine_genres, axis=1)
    movies["final_genres_str"] = movies["final_genres_list"].apply("|".join)

    all_genres = sorted({genre for genres in movies["final_genres_list"] for genre in genres})
    genre2idx = {genre: idx for idx, genre in enumerate(all_genres)}

    def to_multihot(genres: list[str]) -> str:
        arr = np.zeros(len(genre2idx), dtype=np.int8)
        for genre in genres:
            arr[genre2idx[genre]] = 1
        return "[" + " ".join(map(str, arr.tolist())) + "]"

    movies["genre_multihot"] = movies["final_genres_list"].apply(to_multihot)

    keep_columns = [
        "movieId",
        "imdb_id_clean",
        "tmdb_id_clean",
        "title",
        "final_title",
        "original_title",
        "ml_genres_list",
        "poster_genres_list",
        "meta_genres_list",
        "tmdb_all_genres_list",
        "final_genres_list",
        "final_genres_str",
        "poster_url",
        "poster_url_source",
        "poster_url_original",
        "tmdb_all_poster_url",
        "meta_poster_path",
        "poster_imdb_score",
        "meta_vote_average",
        "meta_vote_count",
        "meta_popularity",
        "meta_runtime",
        "release_year",
        "meta_original_language",
        "meta_overview",
        "movie_idx",
        "genre_multihot",
    ]
    movies = movies[keep_columns].sort_values("movie_idx").reset_index(drop=True)

    audit = {
        "movies": len(movies),
        "poster_url_original": int(movies["poster_url_original"].notna().sum()),
        "tmdb_all_poster_url": int(movies["tmdb_all_poster_url"].notna().sum()),
        "poster_url_final": int(movies["poster_url"].notna().sum()),
        "overview": int(movies["meta_overview"].notna().sum()),
        "genres": int(movies["final_genres_str"].astype(str).str.len().gt(0).sum()),
        "poster_source_tmdb_all": int((movies["poster_url_source"] == "tmdb_all").sum()),
        "poster_source_movies_metadata": int(
            (movies["poster_url_source"] == "movies_metadata").sum()
        ),
        "poster_source_moviegenre_imdb": int(
            (movies["poster_url_source"] == "moviegenre_imdb").sum()
        ),
        "poster_source_missing": int((movies["poster_url_source"] == "missing").sum()),
    }
    return movies, audit


def count_csv_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def sample_ratings(
    ratings_path: Path,
    sample_size: int,
    seed: int,
    chunksize: int,
) -> pd.DataFrame:
    total_rows = count_csv_rows(ratings_path)
    if sample_size >= total_rows:
        return pd.read_csv(ratings_path, low_memory=False)

    rng = np.random.default_rng(seed)
    probability = min(1.0, sample_size / total_rows * 1.2)
    parts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(ratings_path, chunksize=chunksize, low_memory=False):
        mask = rng.random(len(chunk)) < probability
        if mask.any():
            parts.append(chunk.loc[mask].copy())
        if sum(len(part) for part in parts) > sample_size * 3:
            sample = pd.concat(parts, ignore_index=True).sample(
                n=sample_size, random_state=seed
            )
            parts = [sample]

    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) < sample_size:
        raise RuntimeError(
            f"Sampling produced only {len(sampled)} rows; expected {sample_size}."
        )
    return sampled.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def make_model_sample(
    ratings_sample: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    feature_columns = [
        "movieId",
        "movie_idx",
        "final_title",
        "final_genres_str",
        "poster_url",
        "poster_url_source",
        "poster_imdb_score",
        "meta_vote_average",
        "meta_vote_count",
        "meta_popularity",
        "meta_runtime",
        "release_year",
        "meta_original_language",
        "meta_overview",
    ]
    return ratings_sample.merge(
        movies[feature_columns],
        on=["movieId", "movie_idx"],
        how="left",
    )


def write_full_model_df(
    ratings_path: Path,
    movies: pd.DataFrame,
    output_path: Path,
    chunksize: int,
) -> None:
    feature_columns = [
        "movieId",
        "movie_idx",
        "final_title",
        "final_genres_str",
        "poster_url",
        "poster_url_source",
        "poster_imdb_score",
        "meta_vote_average",
        "meta_vote_count",
        "meta_popularity",
        "meta_runtime",
        "release_year",
        "meta_original_language",
        "meta_overview",
    ]
    features = movies[feature_columns]
    first = True
    for chunk in pd.read_csv(ratings_path, chunksize=chunksize, low_memory=False):
        out = chunk.merge(features, on=["movieId", "movie_idx"], how="left")
        out.to_csv(output_path, index=False, mode="w" if first else "a", header=first)
        first = False


def write_readme(output_dir: Path, audit: dict, write_full_model_df_enabled: bool) -> None:
    lines = [
        "# Multimodal Movie Dataset v2",
        "",
        "This dataset is rebuilt from the original `MultimodalMovieDataset` plus",
        "`TMDB/TMDB_all_movies.csv` fallback metadata.",
        "",
        "Important changes:",
        "- `poster_url` now prefers TMDB image URLs from `TMDB_all_movies.csv`.",
        "- The old IMDb/Amazon poster URL is preserved as `poster_url_original`.",
        "- `poster_url_source` records where the selected poster URL came from.",
        "- The original `user_idx` and `movie_idx` mappings are preserved.",
        "- Full denormalized `model_df_clean.csv` is optional because it is very large.",
        "",
        "Audit:",
    ]
    lines.extend([f"- {key}: {value}" for key, value in audit.items()])
    lines.extend(
        [
            "",
            "Files:",
            "- `final_movie_features_clean.csv`: one row per movie.",
            "- `final_ratings_clean.csv`: copied rating interactions, if enabled.",
            "- `model_sample_500k.csv`: 500k sampled ratings joined with v2 features.",
            "- `movie2idx.csv`, `user2idx.csv`: unchanged mappings.",
        ]
    )
    if write_full_model_df_enabled:
        lines.append("- `model_df_clean.csv`: full ratings joined with v2 features.")
    else:
        lines.append("- `model_df_clean.csv`: not written by default to avoid a huge file.")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    tmdb_all_path = args.tmdb_all.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building enriched movie features...")
    movies, audit = enrich_movie_features(source_dir, tmdb_all_path)
    movies.to_csv(output_dir / "final_movie_features_clean.csv", index=False)

    print("Copying mappings...")
    shutil.copy2(source_dir / "movie2idx.csv", output_dir / "movie2idx.csv")
    shutil.copy2(source_dir / "user2idx.csv", output_dir / "user2idx.csv")

    ratings_path = source_dir / "final_ratings_clean.csv"
    if not args.no_copy_ratings:
        print("Copying final_ratings_clean.csv...")
        shutil.copy2(ratings_path, output_dir / "final_ratings_clean.csv")

    print(f"Sampling {args.sample_size:,} ratings...")
    ratings_sample = sample_ratings(
        ratings_path,
        sample_size=args.sample_size,
        seed=args.seed,
        chunksize=args.chunksize,
    )
    model_sample = make_model_sample(ratings_sample, movies)
    model_sample.to_csv(output_dir / "model_sample_500k.csv", index=False)

    if args.write_full_model_df:
        print("Writing full model_df_clean.csv. This can take a while...")
        write_full_model_df(
            ratings_path,
            movies,
            output_dir / "model_df_clean.csv",
            chunksize=args.chunksize,
        )

    sample_poster_coverage = int(model_sample["poster_url"].notna().sum())
    sample_unique_movies = model_sample["movie_idx"].nunique()
    sample_movies = movies[movies["movie_idx"].isin(model_sample["movie_idx"].unique())]
    audit.update(
        {
            "sample_rows": len(model_sample),
            "sample_unique_movies": int(sample_unique_movies),
            "sample_rows_with_poster_url": sample_poster_coverage,
            "sample_unique_movies_with_poster_url": int(
                sample_movies["poster_url"].notna().sum()
            ),
        }
    )
    write_readme(output_dir, audit, args.write_full_model_df)

    pd.Series(audit, name="value").to_csv(output_dir / "audit_report.csv")

    print("Done.")
    for key, value in audit.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
