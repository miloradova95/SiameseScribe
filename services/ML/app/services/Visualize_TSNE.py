"""
Standalone t-SNE visualisation for SiameseScribe patch embeddings.

Fetches embeddings directly from ChromaDB (no model inference needed) and
plots a 2D scatter coloured by group or codex label.

Usage (from repo root, conda env pocmedialab):
    python -m services.ML.app.services.Visualize_TSNE
    python -m services.ML.app.services.Visualize_TSNE --n_samples 3000 --color_by codex
    python -m services.ML.app.services.Visualize_TSNE --perplexity 50 --no_show
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[3]
sys.path.append(str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from services.ML.app.chroma_client import get_chroma_client, get_or_create_collection

METADATA_TRAIN = PROJECT_ROOT / "data" / "patches" / "patches_train_metadata.csv"
CHROMA_PATH    = str(PROJECT_ROOT / "data" / "chromaDB" / "data" / "chroma_store")
OUTPUT_DIR     = PROJECT_ROOT / "data" / "models" / "eval_results"


def _stratified_sample(df: pd.DataFrame, n: int, col: str = "group") -> pd.DataFrame:
    """Sample n rows stratified by col; returns all rows if n >= len(df)."""
    if n >= len(df):
        return df
    n_groups = df[col].nunique()
    per_group = max(1, n // n_groups)
    parts = []
    for grp_val in df[col].unique():
        grp_df = df[df[col] == grp_val]
        parts.append(grp_df.sample(min(per_group, len(grp_df)), random_state=42))
    sampled = pd.concat(parts).reset_index(drop=True)
    # top up to exactly n if stratification left us short
    if len(sampled) < n:
        remainder = df[~df["patch_filename"].isin(sampled["patch_filename"])]
        extra = remainder.sample(min(n - len(sampled), len(remainder)), random_state=42)
        sampled = pd.concat([sampled, extra]).reset_index(drop=True)
    return sampled.head(n)


def main():
    parser = argparse.ArgumentParser(description="t-SNE visualisation of patch embeddings")
    parser.add_argument("--collection", default="patches_v1", help="ChromaDB collection name")
    parser.add_argument("--n_samples", type=int, default=2000,
                        help="Number of patches to plot, stratified by group (default: 2000)")
    parser.add_argument("--color_by", choices=["group", "codex"], default="group",
                        help="Metadata field to colour points by (default: group)")
    parser.add_argument("--perplexity", type=float, default=30.0,
                        help="t-SNE perplexity — try 15–50; higher = broader clusters (default: 30)")
    parser.add_argument("--no_show", action="store_true",
                        help="Save the plot without opening an interactive window")
    args = parser.parse_args()

    # ── Load metadata and stratified sample ─────────────────────────────────
    if not METADATA_TRAIN.exists():
        print(f"Metadata CSV not found: {METADATA_TRAIN}")
        sys.exit(1)

    df = pd.read_csv(METADATA_TRAIN)
    df = _stratified_sample(df, args.n_samples, col="group")
    patch_ids = df["patch_filename"].tolist()
    labels    = df[args.color_by].tolist()

    print(f"Sampled {len(patch_ids)} patches (stratified by group), colour_by={args.color_by}")

    # ── Fetch embeddings from ChromaDB ──────────────────────────────────────
    client     = get_chroma_client(CHROMA_PATH)
    collection = get_or_create_collection(client, args.collection)

    print(f"Fetching embeddings from '{args.collection}'...")
    result = collection.get(ids=patch_ids, include=["embeddings"])

    fetched_ids  = result["ids"]
    fetched_embs = result["embeddings"]

    if len(fetched_embs) == 0:
        print("No embeddings returned — run Embedd.py first.")
        sys.exit(1)

    # Align labels to fetched order (some IDs may be absent from ChromaDB)
    id_to_label = dict(zip(patch_ids, labels))
    paired = [(emb, id_to_label[fid]) for fid, emb in zip(fetched_ids, fetched_embs) if fid in id_to_label]
    embs_aligned, labels_aligned = zip(*paired)

    X = np.array(embs_aligned, dtype=np.float32)
    print(f"Embeddings ready: {X.shape[0]} × {X.shape[1]}")

    # ── t-SNE ───────────────────────────────────────────────────────────────
    print(f"Running t-SNE (perplexity={args.perplexity})... this takes ~30–90 s for 2k points")
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        max_iter=1000,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(X)  # (N, 2)

    # ── Plot ────────────────────────────────────────────────────────────────
    unique_labels = sorted(set(labels_aligned))
    cmap = plt.get_cmap("tab10" if len(unique_labels) <= 10 else "tab20")
    color_map = {lbl: cmap(i / max(len(unique_labels) - 1, 1)) for i, lbl in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(10, 8))

    for lbl in unique_labels:
        mask = np.array([l == lbl for l in labels_aligned])
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            color=color_map[lbl],
            label=lbl,
            s=6,
            alpha=0.7,
            linewidths=0,
        )

    ax.legend(title=args.color_by, markerscale=3, fontsize=9, title_fontsize=10, loc="best")
    ax.set_title(
        f"t-SNE  —  {X.shape[0]} patches,  coloured by {args.color_by}\n"
        f"collection={args.collection},  perplexity={args.perplexity}",
        fontsize=12,
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"tsne_{args.color_by}_{timestamp}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
