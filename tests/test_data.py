"""The product-disjoint split, the passage text, the shared smoke slice and the training rows, on synthetic frames."""

import json

import numpy as np
import pandas as pd
import pytest

from esci import data


def judgements(rows):
    """rows: (query_id, split, product_id, label)."""
    return pd.DataFrame([{"query_id": q, "query": f"query {q}", "split": s, "product_id": p, "esci_label": lab} for q, s, p, lab in rows])


EXAMPLES = judgements([
    (1, "train", "p1", "E"), (1, "train", "p2", "S"),
    (2, "train", "p3", "E"), (2, "train", "p5", "I"),     # p5 is judged for test query 4: that judgement must go
    (3, "train", "p6", "E"),                               # p6 is judged for test query 4: query 3 loses its only Exact product
    (4, "test", "p5", "E"), (4, "test", "p6", "S"), (4, "test", "p7", "I"),
    (5, "test", "p8", "S"),                                # no Exact product: cannot be scored on recall
    (6, "train", "p9", "S"),                               # no Exact product
])


def test_split_is_product_disjoint_and_counts_what_it_dropped():
    train, dev, test, stats = data.split_judgements(EXAMPLES, dev_queries=1, seed=42)

    assert not set(train["product_id"]) & set(test["product_id"])
    assert not set(dev["product_id"]) & set(test["product_id"])
    assert set(test["query_id"]) == {4}
    assert set(train["query_id"]) | set(dev["query_id"]) == {1, 2}
    assert not set(train["query_id"]) & set(dev["query_id"])
    assert len(dev["query_id"].unique()) == 1

    assert stats["queries_in_slice"] == 6 and stats["judgements_in_slice"] == 10 and stats["products_in_slice"] == 8
    assert stats["queries_dropped_no_exact"] == 2
    assert stats["train_queries_official"] == 3 and stats["train_judgements_official"] == 5
    assert stats["train_judgements_removed_product_overlap"] == 2
    assert stats["train_products_removed_product_overlap"] == 2
    assert stats["train_queries_dropped_after_overlap_removal"] == 0   # query 3 lost its only judgement, so it is not counted here (documented)
    assert stats["train_queries_official"] - (train["query_id"].nunique() + dev["query_id"].nunique()) == 1
    assert stats["test_products_seen_in_final_train"] == 0
    assert stats["test"] == {"queries": 1, "judgements": 3, "products": 3, "exact_pairs": 1, "exact_per_query_mean": 1.0,
                             "exact_per_query_median": 1.0, "label_share": {"E": 0.333, "S": 0.333, "C": 0.0, "I": 0.333}}


def test_split_is_deterministic_for_a_seed():
    a = data.split_judgements(EXAMPLES, dev_queries=1, seed=7)
    b = data.split_judgements(EXAMPLES, dev_queries=1, seed=7)
    assert a[1]["query_id"].tolist() == b[1]["query_id"].tolist()


def test_product_text_collapses_whitespace_and_skips_empty_fields():
    assert data.product_text("  A   B ", "Acme", None, "x  y\nz") == "A B\nBrand: Acme\nx y z"
    assert data.product_text("A", "  ", "red", None) == "A\nColor: red"
    assert data.product_text(float("nan"), None, None, "bullet") == "\nbullet"     # a missing title keeps the line structure
    assert data.product_text("A B C", None, None, None, max_chars=3) == "A B"


def _write_benchmark(d, n_products=50, n_train=20, n_test=10):
    d.mkdir(parents=True)
    pids = [f"p{i}" for i in range(n_products)]
    pd.DataFrame({"product_id": pids, "text": [f"text {p}" for p in pids]}).to_parquet(d / "corpus.parquet", index=False)
    # test query i (1..n_test) has one Exact product among the last n_test products; train query j has one among the first ones
    judgements([(i, "test", pids[-i], "E") for i in range(1, n_test + 1)]).drop(columns="split").to_parquet(d / "test_judgements.parquet", index=False)
    judgements([(100 + j, "train", pids[j], "E") for j in range(n_train)]).drop(columns="split").to_parquet(d / "train_judgements.parquet", index=False)
    judgements([(200, "train", pids[n_train], "E")]).drop(columns="split").to_parquet(d / "dev_judgements.parquet", index=False)
    (d / "dev_corpus_ids.json").write_text(json.dumps(pids[:5]))
    (d / "stats.json").write_text(json.dumps({"locale": "xx"}))


def test_smoke_slice_does_not_depend_on_the_training_cap(tmp_path):
    """BM25 never passes a training cap, the training CLI may: both must still score the same corpus and test queries."""
    _write_benchmark(tmp_path / "bench")
    a = data.load_benchmark(tmp_path / "bench", corpus_limit=20, max_test_queries=5, max_train_queries=0, seed=3)
    b = data.load_benchmark(tmp_path / "bench", corpus_limit=20, max_test_queries=5, max_train_queries=7, seed=3)
    assert a["test"]["query_id"].tolist() == b["test"]["query_id"].tolist()
    assert a["eval_corpus"]["product_id"].tolist() == b["eval_corpus"]["product_id"].tolist()
    assert len(a["eval_corpus"]) == 20 and set(a["test"]["product_id"]) <= set(a["eval_corpus"]["product_id"])
    assert a["train"]["query_id"].nunique() == 20 and b["train"]["query_id"].nunique() == 7
    assert a["slice"] == {"corpus_limit": 20, "max_test_queries": 5, "max_train_queries": 0, "seed": 3,
                          "eval_corpus_products": 20, "test_queries": 5, "train_queries": 20}


def test_no_limits_means_the_whole_benchmark(tmp_path):
    _write_benchmark(tmp_path / "bench")
    d = data.load_benchmark(tmp_path / "bench")
    assert len(d["eval_corpus"]) == 50 and d["test"]["query_id"].nunique() == 10 and d["stats"] == {"locale": "xx"}


@pytest.fixture
def train_data():
    corpus = pd.DataFrame({"product_id": ["p1", "p2", "p3", "p4", "p9"], "text": ["t1", "t2", "t3", "t4", "t9"]})
    train = judgements([(1, "train", "p1", "E"), (1, "train", "p2", "S"), (1, "train", "p3", "C"), (2, "train", "p4", "E")])
    return {"corpus": corpus, "train": train}


def test_training_examples_by_negative_kind(train_data):
    rows, info = data.training_examples(train_data, "labelled", np.random.default_rng(0))
    assert info["rows"] == 2 and info["queries"] == 2 and info["columns"] == ["anchor", "positive", "negative"]
    by_q = {r["anchor"]: r for r in rows}
    assert by_q["query 1"]["positive"] == "t1" and by_q["query 1"]["negative"] in {"t2", "t3"}
    assert by_q["query 2"]["negative"] != "t4" and info["random_fallback_rows"] == 1   # query 2 has no labelled negative

    rows, _ = data.training_examples(train_data, "substitutes", np.random.default_rng(0))
    assert {r["anchor"]: r["negative"] for r in rows}["query 1"] == "t2"

    rows, info = data.training_examples(train_data, "none", np.random.default_rng(0))
    assert all("negative" not in r for r in rows) and info["columns"] == ["anchor", "positive"]

    rows, _ = data.training_examples(train_data, "mined", np.random.default_rng(0), mined={1: ["p9"], 2: []})
    assert {r["anchor"]: r["negative"] for r in rows}["query 1"] == "t9"


def test_queries_of_is_one_sorted_row_per_query():
    df = judgements([(2, "train", "a", "E"), (1, "train", "b", "E"), (2, "train", "c", "S")])
    assert data.queries_of(df).to_dict("list") == {"query_id": [1, 2], "query": ["query 1", "query 2"]}
