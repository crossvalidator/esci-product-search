"""Retrieval metrics for graded judgements, shared by every retriever in this project (BM25, dense, hybrid).

Inputs are a ranking per query (an ordered list of product ids) and the ESCI judgements per query
(product id → E/S/C/I). Products the ranking returns that were never judged for that query count as
not relevant; this is the standard "pooled" convention (MS MARCO, TREC) and it understates every system
a little, equally.

Metrics (all averaged over queries):
- recall@k   share of the query's Exact products found in the top k. The stage-1 headline: the reranker
             can only reorder what it is handed, so this is what the embedding model is for.
- hit@k      1 if at least one Exact product is in the top k, else 0 (also called accuracy@k / success@k).
- mrr@k      Mean Reciprocal Rank: 1 / rank of the first Exact product (0 if none in the top k).
- ndcg@k     normalized Discounted Cumulative Gain with the KDD Cup 2022 gains E=1.0, S=0.1, C=0.01, I=0;
             credits several relevant products and a partial credit for substitutes. Belongs to the
             reranker and the final list; kept here as a diagnostic.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
CUTOFFS = (10, 20, 50)


def qrels_from_frame(df: pd.DataFrame) -> dict[int, dict[str, str]]:
    """{query_id: {product_id: label}} from a judgements frame."""
    out: dict[int, dict[str, str]] = defaultdict(dict)
    for qid, pid, label in zip(df["query_id"].to_numpy(), df["product_id"].to_numpy(), df["esci_label"].to_numpy(), strict=True):
        out[int(qid)][str(pid)] = str(label)
    return dict(out)


def evaluate(ranked: dict[int, list[str]], qrels: dict[int, dict[str, str]], ks=CUTOFFS) -> dict:
    """Returns {"metrics": {name: mean}, "per_query": DataFrame}. Queries missing from `ranked` score 0."""
    rows = []
    for qid, labels in qrels.items():
        exact = {p for p, lab in labels.items() if lab == "E"}
        top = ranked.get(qid, [])
        first = next((i + 1 for i, p in enumerate(top) if p in exact), None)
        ideal_gains = sorted((GAINS[lab] for lab in labels.values()), reverse=True)
        row = {"query_id": qid, "n_exact": len(exact), "n_judged": len(labels), "first_exact_rank": first}
        for k in ks:
            topk = top[:k]
            hits = sum(p in exact for p in topk)
            row[f"recall@{k}"] = hits / len(exact) if exact else 0.0
            row[f"hit@{k}"] = float(hits > 0)
            row[f"mrr@{k}"] = 1.0 / first if first is not None and first <= k else 0.0
            dcg = sum(GAINS.get(labels.get(p, "I"), 0.0) / math.log2(i + 2) for i, p in enumerate(topk))
            idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains[:k]))
            row[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0
        rows.append(row)
    per_query = pd.DataFrame(rows)
    metric_cols = [c for c in per_query.columns if "@" in c]
    metrics = {c: round(float(per_query[c].mean()), 4) for c in metric_cols}
    metrics["queries"] = int(len(per_query))
    return {"metrics": metrics, "per_query": per_query}


def rrf(rankings: list[dict[int, list[str]]], k: int = 60, depth: int = 100) -> dict[int, list[str]]:
    """Reciprocal Rank Fusion: score(p) = Σ_systems 1 / (k + rank_system(p)). Rank-based, so BM25 scores
    and cosine similarities never have to be put on one scale; the standard first thing to try for hybrid search."""
    qids = set().union(*(r.keys() for r in rankings))
    fused = {}
    for qid in qids:
        score: dict[str, float] = defaultdict(float)
        for r in rankings:
            for i, p in enumerate(r.get(qid, [])[:depth]):
                score[p] += 1.0 / (k + i + 1)
        fused[qid] = [p for p, _ in sorted(score.items(), key=lambda kv: -kv[1])[:depth]]
    return fused


def rankings_to_frame(ranked: dict[int, list[str]], scores: dict[int, list[float]] | None = None) -> pd.DataFrame:
    rows = []
    for qid, pids in ranked.items():
        sc = scores.get(qid) if scores else None
        rows.extend({"query_id": qid, "rank": i + 1, "product_id": p, "score": (sc[i] if sc is not None else np.nan)} for i, p in enumerate(pids))
    return pd.DataFrame(rows, columns=["query_id", "rank", "product_id", "score"])


def save_rankings(path: Path, ranked: dict[int, list[str]], scores: dict[int, list[float]] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rankings_to_frame(ranked, scores).to_parquet(path, index=False)


def load_rankings(path: Path) -> dict[int, list[str]]:
    df = pd.read_parquet(path).sort_values(["query_id", "rank"])
    return {int(q): g["product_id"].tolist() for q, g in df.groupby("query_id", sort=False)}
