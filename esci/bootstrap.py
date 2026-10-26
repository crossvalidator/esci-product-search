"""Paired bootstrap confidence intervals over the frozen test queries.

Every system was scored on the same test queries and its per-query metrics are saved next to its rankings
(`rankings/<label>.per_query.parquet`). A bootstrap resample draws the queries with replacement (the same draw
for every system, hence "paired"), averages the saved per-query scores, and repeats. The spread over resamples is
the uncertainty due to which queries ended up in the test set; no model is re-run.

    uv run python -m esci.bootstrap            # prints the tables and writes outputs/bootstrap.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("outputs")

# label → per-query file (relative to outputs/)
STAGE1 = {
    "BM25": "baselines-us-small-bm25/rankings/bm25.per_query.parquet",
    "bge-base, off the shelf": "baselines-us-small/rankings/BAAI__bge-base-en-v1.5.per_query.parquet",
    "hybrid: BM25 + bge-base": "baselines-us-small-bm25/rankings/hybrid(bm25+BAAI__bge-base-en-v1.5).per_query.parquet",
    "bge-large, off the shelf": "baselines-us-small/rankings/BAAI__bge-large-en-v1.5.per_query.parquet",
    "Qwen3-Embedding-0.6B": "compare-qwen/rankings/Qwen__Qwen3-Embedding-0.6B.per_query.parquet",
    "Qwen3-Embedding-8B": "compare-qwen/rankings/Qwen__Qwen3-Embedding-8B.per_query.parquet",
    "fine-tuned, pairs only": "embed-v1-none/rankings/after.per_query.parquet",
    "fine-tuned, random negatives": "embed-v1-random/rankings/after.per_query.parquet",
    "fine-tuned, mined negatives": "embed-v1-mined/rankings/after.per_query.parquet",
    "fine-tuned, Substitute negatives": "embed-v1-substitutes/rankings/after.per_query.parquet",
    "fine-tuned, labelled negatives": "embed-v1-labelled/rankings/after.per_query.parquet",
    "hybrid: BM25 + fine-tuned": "embed-v1-substitutes-hybrid/rankings/hybrid(bm25+after).per_query.parquet",
}
STAGE2 = {
    "stage-1 order, no reranker": "rerank-offtheshelf/rankings/stage-1 order (no reranker).per_query.parquet",
    "MiniLM reranker, off the shelf": "rerank-offtheshelf/rankings/cross-encoder__ms-marco-MiniLM-L-6-v2.per_query.parquet",
    "bge-reranker-base, off the shelf": "rerank-v1/rankings/before.per_query.parquet",
    "bge-reranker-base, fine-tuned": "rerank-v1/rankings/after.per_query.parquet",
}


def load_matrix(files: dict[str, str], metric: str, root: Path = OUT) -> pd.DataFrame:
    """queries × systems matrix of one per-query metric; only systems whose file exists. Every system was scored on the
    same frozen test queries, so a query missing from one system means a run on a different slice, and the function
    refuses rather than silently dropping queries from every system's interval."""
    cols = {}
    for label, rel in files.items():
        p = root / rel
        if p.exists():
            cols[label] = pd.read_parquet(p).set_index("query_id")[metric]
    mat = pd.DataFrame(cols)
    missing = mat.isna().any(axis=1)
    if missing.any():
        short = {c: int(mat[c].isna().sum()) for c in mat.columns if mat[c].isna().any()}
        raise ValueError(f"{int(missing.sum())} of {len(mat)} queries are missing from at least one system (missing per system: {short}); "
                         "the systems must be scored on the same test queries")
    return mat


def bootstrap(mat: pd.DataFrame, n: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Means per system per resample. Returns (summary table, the n × systems array of resampled means)."""
    rng = np.random.default_rng(seed)
    values = mat.to_numpy()
    q = values.shape[0]
    means = np.empty((n, values.shape[1]))
    for i in range(n):
        idx = rng.integers(0, q, size=q)
        means[i] = values[idx].mean(axis=0)
    lo, hi = np.percentile(means, [2.5, 97.5], axis=0)
    summary = pd.DataFrame({"mean": values.mean(axis=0), "ci_low": lo, "ci_high": hi}, index=mat.columns).round(4)
    return summary, means


def differences(mat: pd.DataFrame, means: np.ndarray, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """For each (a, b): the observed difference a − b, its interval, and the share of resamples with a > b."""
    cols = list(mat.columns)
    rows = []
    for a, b in pairs:
        if a not in cols or b not in cols:
            continue
        d = means[:, cols.index(a)] - means[:, cols.index(b)]
        rows.append({"a": a, "b": b, "diff": round(float(mat[a].mean() - mat[b].mean()), 4),
                     "ci_low": round(float(np.percentile(d, 2.5)), 4), "ci_high": round(float(np.percentile(d, 97.5)), 4),
                     "share_a_ahead": round(float((d > 0).mean()), 3)})
    return pd.DataFrame(rows)


STAGE1_PAIRS = [
    ("fine-tuned, labelled negatives", "Qwen3-Embedding-8B"),
    ("hybrid: BM25 + fine-tuned", "Qwen3-Embedding-8B"),
    ("fine-tuned, labelled negatives", "bge-base, off the shelf"),
    ("fine-tuned, labelled negatives", "hybrid: BM25 + bge-base"),
    ("fine-tuned, labelled negatives", "bge-large, off the shelf"),
    ("fine-tuned, labelled negatives", "fine-tuned, random negatives"),
    ("fine-tuned, labelled negatives", "fine-tuned, pairs only"),
    ("fine-tuned, labelled negatives", "fine-tuned, mined negatives"),
    ("fine-tuned, labelled negatives", "fine-tuned, Substitute negatives"),
    ("fine-tuned, mined negatives", "bge-base, off the shelf"),
    ("hybrid: BM25 + fine-tuned", "fine-tuned, Substitute negatives"),
    ("hybrid: BM25 + bge-base", "bge-base, off the shelf"),
    ("hybrid: BM25 + bge-base", "bge-large, off the shelf"),
    ("Qwen3-Embedding-8B", "bge-base, off the shelf"),
    ("bge-base, off the shelf", "Qwen3-Embedding-0.6B"),
    ("BM25", "Qwen3-Embedding-0.6B"),
]
STAGE2_PAIRS = [
    ("bge-reranker-base, fine-tuned", "stage-1 order, no reranker"),
    ("bge-reranker-base, fine-tuned", "MiniLM reranker, off the shelf"),
    ("bge-reranker-base, fine-tuned", "bge-reranker-base, off the shelf"),
    ("MiniLM reranker, off the shelf", "stage-1 order, no reranker"),
    ("bge-reranker-base, off the shelf", "stage-1 order, no reranker"),
]


def run_all(n: int = 2000, seed: int = 0, root: Path = OUT) -> dict:
    out = {"resamples": n, "seed": seed, "tables": {}}
    for name, files, metric, pairs in (("stage1_recall@20", STAGE1, "recall@20", STAGE1_PAIRS), ("stage1_recall@10", STAGE1, "recall@10", STAGE1_PAIRS),
                                       ("stage2_hit@3", STAGE2, "hit@3", STAGE2_PAIRS), ("stage2_ndcg@3", STAGE2, "ndcg@3", STAGE2_PAIRS)):
        mat = load_matrix(files, metric, root)
        if mat.empty:
            continue
        summary, means = bootstrap(mat, n, seed)
        diffs = differences(mat, means, pairs)
        out["tables"][name] = {"queries": int(len(mat)), "metric": metric, "systems": summary.reset_index().rename(columns={"index": "system"}).to_dict("records"),
                               "differences": diffs.to_dict("records")}
    return out


if __name__ == "__main__":
    res = run_all()
    for name, t in res["tables"].items():
        print(f"\n=== {name}: {t['queries']} queries, {res['resamples']} resamples")
        print(pd.DataFrame(t["systems"]).set_index("system").to_string())
        print(pd.DataFrame(t["differences"]).to_string(index=False))
    OUT.mkdir(exist_ok=True)
    (OUT / "bootstrap.json").write_text(json.dumps(res, indent=2))
    print(f"\nwritten to {OUT / 'bootstrap.json'}")
