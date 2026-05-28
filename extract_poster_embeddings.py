"""
Poster embedding extractor.

Downloads poster images and extracts MobileNetV2 embeddings aligned by
movie_idx. By default, the script auto-detects the processed dataset folder
when it is run from the project checkout.

Expected files in DATA_DIR:
  - movie2idx.csv
  - one of: final_movie_features_clean.csv, model_df_clean.csv,
    model_sample_500k.csv

Output:
  - poster_embeddings.npy with shape [N_MOVIES, 1280]
"""

from __future__ import annotations

import argparse
import io
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from zipfile import ZipFile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageOps


MOVIE_CSV_CANDIDATES = (
    "final_movie_features_clean.csv",
    "model_df_clean.csv",
    "model_sample_500k.csv",
)
EMBED_DIM = 1280
IMG_SIZE = (224, 224)
USER_AGENT = "Mozilla/5.0 (compatible; poster-embedding-extractor/1.0)"
INVALID_URL_TOKENS = {"", "nan", "none", "<na>", "null"}


def script_dir() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()


def default_data_dir() -> Path:
    root = script_dir()
    candidates = [
        root / "dataset" / "MultimodalMovieDataset_v2",
        root / "MultimodalMovieDataset_v2",
        root,
    ]
    for candidate in candidates:
        has_movie_csv = any((candidate / name).exists() for name in MOVIE_CSV_CANDIDATES)
        if (candidate / "movie2idx.csv").exists() and has_movie_csv:
            return candidate
    return root


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract poster CNN embeddings.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="Folder containing movie2idx.csv and the movie feature CSV.",
    )
    parser.add_argument(
        "--movie-csv",
        type=Path,
        default=None,
        help="Optional explicit movie feature CSV. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--output-npy",
        type=Path,
        default=None,
        help="Output .npy path. Defaults to DATA_DIR/poster_embeddings.npy.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Poster image cache folder. Defaults to DATA_DIR/poster_cache.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint .npy path. Defaults to DATA_DIR/poster_embeddings_checkpoint.npy.",
    )
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--workers", type=positive_int, default=20)
    parser.add_argument("--save-every", type=positive_int, default=1000)
    parser.add_argument("--timeout", type=positive_int, default=10)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use only images already present in the cache.",
    )
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Redownload/re-encode rows that already have nonzero embeddings.",
    )
    parser.add_argument(
        "--no-unzip-cache",
        action="store_true",
        help="Do not auto-extract poster_cache*.zip when the cache is empty.",
    )
    return parser.parse_args()


def clean_url(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in INVALID_URL_TOKENS else text


def find_movie_csv(data_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit if explicit.is_absolute() else data_dir / explicit
        if not path.exists():
            raise FileNotFoundError(f"Movie CSV not found: {path}")
        return path

    for name in MOVIE_CSV_CANDIDATES:
        path = data_dir / name
        if path.exists():
            return path

    expected = ", ".join(MOVIE_CSV_CANDIDATES)
    raise FileNotFoundError(
        f"No movie feature CSV found in {data_dir}. Expected one of: {expected}"
    )


def load_movie_list(movie_csv: Path, movie2idx_csv: Path) -> tuple[pd.DataFrame, int]:
    if not movie2idx_csv.exists():
        raise FileNotFoundError(f"movie2idx.csv not found: {movie2idx_csv}")

    movie2idx = pd.read_csv(movie2idx_csv, usecols=["movie_idx"])
    n_movies = int(movie2idx["movie_idx"].max()) + 1

    movies = pd.read_csv(movie_csv, usecols=["movie_idx", "poster_url"])
    movies = movies.dropna(subset=["movie_idx"]).copy()
    movies["movie_idx"] = movies["movie_idx"].astype(int)
    movies["poster_url"] = movies["poster_url"].apply(clean_url)
    movies["has_url"] = movies["poster_url"].ne("")

    bad = movies[(movies["movie_idx"] < 0) | (movies["movie_idx"] >= n_movies)]
    if len(bad):
        example = bad["movie_idx"].head(5).tolist()
        raise ValueError(
            f"{len(bad)} rows have movie_idx outside [0, {n_movies - 1}]. "
            f"Examples: {example}"
        )

    movies = (
        movies.sort_values(["movie_idx", "has_url"], ascending=[True, False])
        .drop_duplicates("movie_idx")
        .sort_values("movie_idx")
        .reset_index(drop=True)
    )
    return movies[["movie_idx", "poster_url", "has_url"]], n_movies


def safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    with ZipFile(zip_path) as zf:
        for member in zf.infolist():
            member_path = (target_root / member.filename).resolve()
            if target_root != member_path and target_root not in member_path.parents:
                raise ValueError(f"Unsafe path in zip archive: {member.filename}")
        zf.extractall(target_root)


def maybe_unzip_cache(data_dir: Path, cache_dir: Path) -> None:
    existing = list(cache_dir.glob("*.jpg")) if cache_dir.exists() else []
    if existing:
        return

    zips = sorted(data_dir.glob("poster_cache*.zip"))
    if not zips:
        return

    zip_path = zips[0]
    print(f"    Cache empty. Extracting {zip_path.name}...")
    safe_extract_zip(zip_path, data_dir)


def cached_image_is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:  # noqa: BLE001 - bad cache files are re-downloaded
        return False


def download_one(row: dict, cache_dir: Path, timeout: int) -> tuple[int, str]:
    idx = int(row["movie_idx"])
    url = row["poster_url"]
    dest = cache_dir / f"{idx}.jpg"

    if dest.exists():
        if cached_image_is_valid(dest):
            return idx, "cached"
        try:
            dest.unlink()
        except OSError:
            return idx, "bad_cache"
    if not url:
        return idx, "missing_url"

    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            stream=True,
        )
        if response.status_code != 200:
            return idx, f"http_{response.status_code}"

        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image = ImageOps.pad(image, IMG_SIZE, color=(0, 0, 0))
        image.save(dest, "JPEG", quality=90)
        return idx, "ok"
    except Exception as exc:  # noqa: BLE001 - keep batch job running
        return idx, f"err_{type(exc).__name__}"


def load_or_init_embeddings(
    n_movies: int,
    out_npy: Path,
    checkpoint: Path,
) -> np.ndarray:
    for candidate in (checkpoint, out_npy):
        if not candidate.exists():
            continue
        embeddings = np.load(candidate)
        expected_shape = (n_movies, EMBED_DIM)
        if embeddings.shape != expected_shape:
            print(
                f"    Ignoring {candidate.name}: shape {embeddings.shape} "
                f"!= {expected_shape}"
            )
            continue
        print(f"    Resuming from {candidate}")
        return embeddings.astype(np.float32, copy=False)

    embeddings = np.zeros((n_movies, EMBED_DIM), dtype=np.float32)
    print(f"    Fresh embeddings array: {embeddings.shape}")
    return embeddings


def existing_done_indices(n_movies: int, out_npy: Path, checkpoint: Path) -> set[int]:
    expected_shape = (n_movies, EMBED_DIM)
    for candidate in (checkpoint, out_npy):
        if not candidate.exists():
            continue
        embeddings = np.load(candidate, mmap_mode="r")
        if embeddings.shape == expected_shape:
            return set(np.flatnonzero(np.any(embeddings != 0, axis=1)).tolist())
    return set()


def cache_image_paths(cache_dir: Path, n_movies: int) -> list[Path]:
    paths = []
    for path in cache_dir.glob("*.jpg"):
        try:
            idx = int(path.stem)
        except ValueError:
            continue
        if 0 <= idx < n_movies:
            paths.append(path)
    return sorted(paths, key=lambda p: int(p.stem))


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir.resolve()
    movie_csv = find_movie_csv(data_dir, args.movie_csv)
    movie2idx_csv = data_dir / "movie2idx.csv"
    out_npy = (args.output_npy or data_dir / "poster_embeddings.npy").resolve()
    cache_dir = (args.cache_dir or data_dir / "poster_cache").resolve()
    checkpoint = (
        args.checkpoint or data_dir / "poster_embeddings_checkpoint.npy"
    ).resolve()

    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    out_npy.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  POSTER EMBEDDING EXTRACTOR")
    print("=" * 72)
    print(f"Data dir   : {data_dir}")
    print(f"Movie CSV  : {movie_csv.name}")
    print(f"movie2idx  : {movie2idx_csv.name}")
    print(f"Cache dir  : {cache_dir}")
    print(f"Output npy : {out_npy}")

    if not args.no_unzip_cache:
        maybe_unzip_cache(data_dir, cache_dir)

    print("\n[1] Loading movie list...")
    movies, n_movies = load_movie_list(movie_csv, movie2idx_csv)
    todo_dl = movies[movies["has_url"]].copy()
    print(f"    Total indexed movies : {n_movies:,}")
    print(f"    Movies in CSV        : {len(movies):,}")
    print(f"    Movies with URL      : {len(todo_dl):,}")

    done_before_download = (
        set()
        if args.refresh_existing
        else existing_done_indices(n_movies, out_npy, checkpoint)
    )
    if done_before_download:
        before = len(todo_dl)
        todo_dl = todo_dl[~todo_dl["movie_idx"].isin(done_before_download)].copy()
        skipped = before - len(todo_dl)
        print(f"    Already embedded    : {skipped:,} URL rows skipped")

    if args.skip_download:
        print("\n[2] Download skipped.")
    else:
        print(f"\n[2] Downloading posters ({args.workers} workers)...")
        rows = todo_dl.to_dict("records")
        total = len(rows)
        done = ok = fail = 0
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(download_one, row, cache_dir, args.timeout): row
                for row in rows
            }
            for future in as_completed(futures):
                _, status = future.result()
                done += 1
                if status in {"ok", "cached"}:
                    ok += 1
                else:
                    fail += 1
                if done % 1000 == 0 or done == total:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 1
                    eta = (total - done) / rate / 60
                    print(
                        f"  [{done:>6}/{total}] ok={ok} fail={fail} "
                        f"ETA={eta:.1f} min"
                    )

        print(f"    Download complete: {ok:,} cached/saved, {fail:,} failed")

    print("\n[3] Preparing embedding array...")
    embeddings = load_or_init_embeddings(n_movies, out_npy, checkpoint)
    already_done = set(np.flatnonzero(np.any(embeddings != 0, axis=1)).tolist())
    image_paths = cache_image_paths(cache_dir, n_movies)
    todo_cnn = [p for p in image_paths if int(p.stem) not in already_done]
    print(f"    Images on disk  : {len(image_paths):,}")
    print(f"    Already encoded : {len(already_done):,}")
    print(f"    To process      : {len(todo_cnn):,}")

    if not todo_cnn:
        print("\n[4] No new poster images to encode. TensorFlow was not loaded.")
        print(f"\n[5] Saving final embeddings -> {out_npy}")
        np.save(out_npy, embeddings)
        covered = int(np.any(embeddings != 0, axis=1).sum())
        print(f"    Shape   : {embeddings.shape}")
        print(f"    Covered : {covered:,} / {n_movies:,} movies")
        print("    Done.")
        return

    print("\n[4] Loading TensorFlow + MobileNetV2...")
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(42)
    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False
    preprocess = keras.applications.mobilenet_v2.preprocess_input
    print(f"    MobileNetV2 ready. Output shape: {base.output_shape}")

    print(f"\n[5] Extracting features (batch={args.batch_size})...")
    total = len(todo_cnn)
    done = 0
    t1 = time.time()

    for batch_start in range(0, total, args.batch_size):
        batch_paths = todo_cnn[batch_start : batch_start + args.batch_size]
        arrays, idxs = [], []

        for path in batch_paths:
            try:
                image = Image.open(path).convert("RGB").resize(IMG_SIZE)
                arrays.append(np.asarray(image, dtype=np.float32))
                idxs.append(int(path.stem))
            except Exception as exc:  # noqa: BLE001 - skip broken cache files
                print(f"    Skipping {path.name}: {type(exc).__name__}")

        if arrays:
            batch = preprocess(np.stack(arrays))
            features = base.predict(batch, verbose=0)
            for i, idx in enumerate(idxs):
                embeddings[idx] = features[i]

        done += len(batch_paths)
        if done % args.save_every == 0 or done >= total:
            np.save(checkpoint, embeddings)
            elapsed = time.time() - t1
            rate = done / elapsed if elapsed > 0 else 1
            eta = (total - done) / rate / 60
            pct = done / total * 100 if total else 100
            print(
                f"  [{done:>6}/{total}] {pct:5.1f}% "
                f"speed={rate:.1f}/s ETA={eta:.1f} min [checkpoint]"
            )

    print(f"\n[6] Saving final embeddings -> {out_npy}")
    np.save(out_npy, embeddings)
    covered = int(np.any(embeddings != 0, axis=1).sum())
    print(f"    Shape   : {embeddings.shape}")
    print(f"    Covered : {covered:,} / {n_movies:,} movies")
    print("    Done.")


if __name__ == "__main__":
    main()
