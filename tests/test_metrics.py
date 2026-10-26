"""recall@k, hit@k, MRR@k and graded nDCG@k on a hand-built example; Reciprocal Rank Fusion; ranking files."""

import math

import numpy as np
import pandas as pd
import pytest

from esci import metrics

QRELS = {1: {"a": "E", "b": "E", "c": "S", "d": "I"}, 2: {"x": "E", "y": "C"}}
RANKED = {1: ["c", "a", "z", "b"], 2: ["y", "q"]}   # z and q were never judged for their query


def test_metrics_on_a_worked_example():
    res = metrics.evaluate(RANKED, QRELS, ks=(2, 3))
    pq = res["per_query"].set_index("query_id")

    # query 1: Exact products a, b; the first Exact one is at rank 2
    q1 = pq.loc[1]
    assert q1["n_exact"] == 2 and q1["n_judged"] == 4 and q1["first_exact_rank"] == 2
    assert q1["recall@2"] == 0.5 and q1["hit@2"] == 1.0 and q1["mrr@2"] == 0.5
    assert q1["recall@3"] == 0.5                      # z (unjudged) counts as not relevant
    dcg2 = 0.1 / math.log2(2) + 1.0 / math.log2(3)   # c (Substitute) at rank 1, a (Exact) at rank 2
    idcg2 = 1.0 / math.log2(2) + 1.0 / math.log2(3)  # the two Exact products first
    assert q1["ndcg@2"] == pytest.approx(dcg2 / idcg2)
    idcg3 = idcg2 + 0.1 / math.log2(4)
    assert q1["ndcg@3"] == pytest.approx(dcg2 / idcg3)

    # query 2: its only Exact product is not retrieved at all
    q2 = pq.loc[2]
    assert q2["recall@2"] == 0.0 and q2["hit@2"] == 0.0 and q2["mrr@2"] == 0.0
    assert pd.isna(q2["first_exact_rank"])
    assert q2["ndcg@2"] == pytest.approx((0.01 / math.log2(2)) / (1.0 + 0.01 / math.log2(3)))

    m = res["metrics"]
    assert m["queries"] == 2
    assert m["recall@2"] == 0.25 and m["hit@2"] == 0.5 and m["mrr@2"] == 0.25
    assert m["ndcg@2"] == round(float(pq["ndcg@2"].mean()), 4)


def test_query_missing_from_the_ranking_scores_zero():
    res = metrics.evaluate({}, QRELS, ks=(10,))
    assert res["metrics"]["queries"] == 2
    assert res["metrics"]["recall@10"] == 0.0 and res["metrics"]["ndcg@10"] == 0.0


def test_qrels_from_frame_uses_python_ints_and_strs():
    df = pd.DataFrame({"query_id": np.array([7, 7], dtype=np.int64), "product_id": ["p", "q"], "esci_label": ["E", "I"]})
    qrels = metrics.qrels_from_frame(df)
    assert qrels == {7: {"p": "E", "q": "I"}}
    assert type(next(iter(qrels))) is int


def test_rrf_sums_reciprocal_ranks_across_systems():
    fused = metrics.rrf([{1: ["a", "b", "c"]}, {1: ["c", "a"]}], k=60)
    # a: 1/61 + 1/62; c: 1/63 + 1/61; b: 1/62 -> a ahead of c ahead of b
    assert fused[1] == ["a", "c", "b"]
    assert 1 / 61 + 1 / 62 > 1 / 63 + 1 / 61 > 1 / 62


def test_rrf_respects_depth_and_unions_queries():
    fused = metrics.rrf([{1: ["a", "b", "c"]}, {2: ["x"]}], depth=2)
    assert fused[1] == ["a", "b"] and fused[2] == ["x"]


def test_rankings_round_trip(tmp_path):
    ranked = {2: ["p", "q"], 1: ["r"]}
    scores = {2: [0.9, 0.1], 1: [0.5]}
    path = tmp_path / "rankings" / "x.parquet"
    metrics.save_rankings(path, ranked, scores)
    df = pd.read_parquet(path)
    assert list(df.columns) == ["query_id", "rank", "product_id", "score"]
    assert df["rank"].tolist() == [1, 2, 1]
    assert metrics.load_rankings(path) == ranked
    assert metrics.rankings_to_frame(ranked)["score"].isna().all()


def test_bm25_ranking_is_tie_broken_deterministically():
    from esci import baselines

    pids, scores = ["b", "a", "c", "d"], [1.0, 1.0 + 1e-12, 0.5, 2.0]
    assert baselines.stable_order(pids, scores) == (["d", "a", "b", "c"], [2.0, 1.0 + 1e-12, 1.0, 0.5])


def test_bm25_on_a_tiny_corpus_is_stable_across_calls():
    from esci import baselines

    corpus = ["red shoes for running", "running shoes red", "blue hat", "red hat", "shoes"]
    ids = [f"p{i}" for i in range(5)]
    queries = [(1, "red shoes"), (2, "hat")]
    a = baselines.bm25_rankings(corpus, ids, queries, k=5)
    b = baselines.bm25_rankings(corpus, ids, queries, k=5)
    assert a[0] == b[0] and a[1] == b[1]
    assert a[0][2][0] in {"p2", "p3"} and set(a[0][1][:2]) == {"p0", "p1"}
