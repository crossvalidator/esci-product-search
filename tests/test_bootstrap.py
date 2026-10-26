"""Paired bootstrap on a tiny matrix, and the refusal to mix systems scored on different queries."""

import pandas as pd
import pytest

from esci import bootstrap


def test_bootstrap_summary_and_paired_differences():
    mat = pd.DataFrame({"A": [1.0, 1.0, 1.0, 0.0], "B": [0.0, 0.0, 0.0, 1.0]})
    summary, means = bootstrap.bootstrap(mat, n=300, seed=1)
    assert means.shape == (300, 2)
    assert summary.loc["A", "mean"] == 0.75 and summary.loc["B", "mean"] == 0.25
    assert summary.loc["A", "ci_low"] <= 0.75 <= summary.loc["A", "ci_high"]
    assert (means.sum(axis=1) == 1.0).all()           # paired: the same resample of queries for both systems

    diffs = bootstrap.differences(mat, means, [("A", "B"), ("A", "missing")])
    assert len(diffs) == 1                            # pairs with an unknown system are skipped
    assert diffs.loc[0, "diff"] == 0.5
    assert 0.6 < diffs.loc[0, "share_a_ahead"] < 0.85   # A is ahead when at least 3 of 4 draws hit its three winning rows: Binomial(4, 0.75) tail, 0.74


def test_bootstrap_is_seeded():
    mat = pd.DataFrame({"A": [0.2, 0.4, 0.9], "B": [0.1, 0.5, 0.3]})
    assert (bootstrap.bootstrap(mat, n=50, seed=5)[1] == bootstrap.bootstrap(mat, n=50, seed=5)[1]).all()


def _per_query(root, rel, qids):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"query_id": qids, "recall@20": [0.5] * len(qids)}).to_parquet(p, index=False)


def test_load_matrix_keeps_only_existing_files_and_refuses_mismatched_queries(tmp_path):
    files = {"one": "r1/rankings/x.per_query.parquet", "two": "r2/rankings/y.per_query.parquet", "absent": "nope.parquet"}
    _per_query(tmp_path, files["one"], [1, 2, 3])
    _per_query(tmp_path, files["two"], [1, 2, 3])
    mat = bootstrap.load_matrix(files, "recall@20", root=tmp_path)
    assert list(mat.columns) == ["one", "two"] and len(mat) == 3

    _per_query(tmp_path, files["two"], [1, 2])
    with pytest.raises(ValueError, match="1 of 3 queries are missing"):
        bootstrap.load_matrix(files, "recall@20", root=tmp_path)


def test_run_all_with_no_outputs_has_no_tables(tmp_path):
    assert bootstrap.run_all(n=10, root=tmp_path)["tables"] == {}
