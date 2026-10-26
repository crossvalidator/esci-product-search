"""Turn the Amazon Shopping Queries Dataset (ESCI) into a retrieval benchmark.

ESCI ships as two parquet files: `examples` (one row per judged query-product pair, with the label
E/S/C/I = Exact / Substitute / Complement / Irrelevant, the locale, the official train/test split
and two flags `small_version` / `large_version`) and `products` (one row per product and locale:
title, description, bullet points, brand, colour). The "small" and "large" versions are flags on
the same file, not separate downloads; the small version is the harder subset used for the KDD Cup
2022 ranking task.

This module does three things, each a plain function so it can run on a Modal container or locally:

1. `download()`            fetch the raw files (the examples file is 51 MB, products 1.1 GB).
2. `split_judgements()`    slice to one locale and version, drop queries without an Exact product,
                           make the training data product-disjoint from the test queries, and carve
                           a small dev set out of the training queries. Needs only the examples file.
3. `build()`               do 2, then attach product text and write the benchmark files:
                           corpus.parquet, train/dev/test_judgements.parquet, dev_corpus_ids.json, stats.json.

`load_benchmark()` reads those files back and applies the optional smoke-slice limits that every
evaluation script shares, so BM25, dense and hybrid results are always computed on identical slices.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

LFS = "https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset/"
RAW = "https://raw.githubusercontent.com/amazon-science/esci-data/main/shopping_queries_dataset/"
RAW_FILES = {
    # name: (url, size in bytes, sha256) — sizes/hashes from the repository's Git LFS pointers
    "examples": (LFS + "shopping_queries_dataset_examples.parquet", 51_286_808,
                 "4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a"),
    "products": (LFS + "shopping_queries_dataset_products.parquet", 1_108_857_465,
                 "25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265"),
    "sources": (RAW + "shopping_queries_dataset_sources.csv", None, None),
}
LABELS = ("E", "S", "C", "I")
NEGATIVES = ("labelled", "substitutes", "mined", "random", "none")
PASSAGE_MAX_CHARS = 1200   # ~250 tokens: the title, brand/colour line and the start of the bullets


def raw_path(raw_dir: Path, name: str) -> Path:
    return Path(raw_dir) / RAW_FILES[name][0].rsplit("/", 1)[1]


def download(raw_dir: Path, names=("examples", "sources"), log=print) -> dict[str, Path]:
    """Fetch raw files into raw_dir, skipping files already present with the right size. Verifies sha256."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in names:
        url, size, sha = RAW_FILES[name]
        dest = raw_path(raw_dir, name)
        marker = dest.with_name(dest.name + ".sha256")   # written once the hash has been checked: the 1.1 GB file is not re-hashed on every call
        if dest.exists() and (size is None or dest.stat().st_size == size):
            log(f"{dest.name}: already present ({dest.stat().st_size / 1e6:.0f} MB)")
        else:
            marker.unlink(missing_ok=True)
            t0, done, mark = time.time(), 0, 0
            with urllib.request.urlopen(url) as r, open(dest, "wb") as fh:
                while chunk := r.read(8 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if done - mark >= 100e6:
                        mark = done
                        log(f"{dest.name}: {done / 1e6:.0f} MB in {time.time() - t0:.0f}s")
            log(f"{dest.name}: downloaded {done / 1e6:.0f} MB in {time.time() - t0:.0f}s")
        if sha and not (marker.exists() and marker.read_text().strip() == sha):
            h = hashlib.sha256()
            with open(dest, "rb") as fh:
                while chunk := fh.read(8 << 20):
                    h.update(chunk)
            if h.hexdigest() != sha:
                raise RuntimeError(f"{dest.name}: sha256 mismatch, delete and retry")
            marker.write_text(sha)
        out[name] = dest
    return out


def load_examples(raw_dir: Path, locale: str = "us", version: str = "small") -> pd.DataFrame:
    ex = pd.read_parquet(raw_path(raw_dir, "examples"))
    ex = ex[ex["product_locale"] == locale]
    if version == "small":
        ex = ex[ex["small_version"] == 1]
    elif version != "large":
        raise ValueError("version must be 'small' or 'large'")
    return ex.reset_index(drop=True)


def split_judgements(ex: pd.DataFrame, dev_queries: int = 500, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Official query split, made product-disjoint: every product that appears in any test query's judged
    list is removed from the training judgements, so no test product (relevant or not) is ever seen in training.
    Then `dev_queries` training queries are held out for model selection during training.
    `train_queries_dropped_after_overlap_removal` counts the training queries that kept some judgement but lost their last
    Exact one; a query whose every judgement was removed is not in it (the Spark build in notebook 04 reports the full number,
    official minus kept, and the two are reconciled there)."""
    cols = ["query_id", "query", "product_id", "esci_label"]
    stats: dict = {"queries_in_slice": int(ex["query_id"].nunique()), "judgements_in_slice": int(len(ex)),
                   "products_in_slice": int(ex["product_id"].nunique())}
    has_exact = set(ex.loc[ex["esci_label"] == "E", "query_id"])
    stats["queries_dropped_no_exact"] = int(ex["query_id"].nunique() - len(has_exact))
    ex = ex[ex["query_id"].isin(has_exact)]

    test = ex[ex["split"] == "test"][cols]
    train = ex[ex["split"] == "train"][cols]
    stats["train_queries_official"], stats["train_judgements_official"] = int(train["query_id"].nunique()), int(len(train))
    test_products = set(test["product_id"])
    leaked = train["product_id"].isin(test_products)
    stats["train_judgements_removed_product_overlap"] = int(leaked.sum())
    stats["train_products_removed_product_overlap"] = int(train.loc[leaked, "product_id"].nunique())
    train = train[~leaked]
    still_has_exact = set(train.loc[train["esci_label"] == "E", "query_id"])
    stats["train_queries_dropped_after_overlap_removal"] = int(train["query_id"].nunique() - len(still_has_exact))
    train = train[train["query_id"].isin(still_has_exact)]

    rng = np.random.default_rng(seed)
    qids = np.sort(train["query_id"].unique())
    dev_ids = set(rng.choice(qids, size=min(dev_queries, len(qids)), replace=False).tolist())
    dev = train[train["query_id"].isin(dev_ids)]
    train = train[~train["query_id"].isin(dev_ids)]

    for name, df in (("train", train), ("dev", dev), ("test", test)):
        e = df[df["esci_label"] == "E"].groupby("query_id").size()
        stats[name] = {"queries": int(df["query_id"].nunique()), "judgements": int(len(df)), "products": int(df["product_id"].nunique()),
                       "exact_pairs": int(len(e) and e.sum()), "exact_per_query_mean": round(float(e.mean()), 2), "exact_per_query_median": float(e.median()),
                       "label_share": {lab: round(float((df["esci_label"] == lab).mean()), 3) for lab in LABELS}}
    stats["test_products_seen_in_final_train"] = int(len(set(test["product_id"]) & set(train["product_id"])))
    return train.reset_index(drop=True), dev.reset_index(drop=True), test.reset_index(drop=True), stats


def product_text(title, brand, color, bullets, max_chars: int = PASSAGE_MAX_CHARS) -> str:
    """What the embedding model sees for one product: title, a brand/colour line, then the bullet points,
    whitespace-collapsed and cut at max_chars. The description is left out in v1: it is long, often HTML,
    and mostly repeats the bullets."""
    parts = [" ".join(str(title).split())] if isinstance(title, str) else [""]
    meta = ", ".join(f"{k}: {' '.join(str(v).split())}" for k, v in (("Brand", brand), ("Color", color)) if isinstance(v, str) and v.strip())
    if meta:
        parts.append(meta)
    if isinstance(bullets, str) and bullets.strip():
        parts.append(" ".join(bullets.split()))
    return "\n".join(parts)[:max_chars]


def build(raw_dir: Path, out_dir: Path, locale: str = "us", version: str = "small", dev_queries: int = 500,
          dev_corpus_size: int = 20_000, seed: int = 42, log=print) -> dict:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ex = load_examples(raw_dir, locale, version)
    train, dev, test, stats = split_judgements(ex, dev_queries, seed)
    stats.update({"locale": locale, "version": version, "seed": seed, "passage_max_chars": PASSAGE_MAX_CHARS})
    log(f"judgements split in {time.time() - t0:.0f}s: train {stats['train']['queries']} q, dev {stats['dev']['queries']} q, test {stats['test']['queries']} q")

    corpus_ids = pd.Index(ex["product_id"].unique())
    products = pd.read_parquet(raw_path(raw_dir, "products"),
                               columns=["product_id", "product_title", "product_brand", "product_color", "product_bullet_point"],
                               filters=[("product_locale", "==", locale)])
    products = products[products["product_id"].isin(corpus_ids)].drop_duplicates("product_id").reset_index(drop=True)
    stats["corpus_products_missing_from_products_file"] = int(len(corpus_ids) - len(products))
    products["text"] = [product_text(t, b, c, bp) for t, b, c, bp in
                        zip(products["product_title"], products["product_brand"], products["product_color"], products["product_bullet_point"], strict=True)]
    corpus = products.rename(columns={"product_title": "title", "product_brand": "brand", "product_color": "color"})[
        ["product_id", "title", "brand", "color", "text"]]
    corpus["n_chars"] = corpus["text"].str.len().astype("int32")
    stats["corpus"] = {"products": int(len(corpus)), "text_chars_mean": round(float(corpus["n_chars"].mean()), 1),
                       "text_chars_median": float(corpus["n_chars"].median()),
                       "share_cut_at_max_chars": round(float((corpus["n_chars"] >= PASSAGE_MAX_CHARS).mean()), 3),
                       "share_with_brand": round(float(corpus["brand"].notna().mean()), 3),
                       "share_with_bullets": round(float(products["product_bullet_point"].notna().mean()), 3)}
    log(f"corpus: {len(corpus)} products, text built in {time.time() - t0:.0f}s")

    # The in-training evaluator re-encodes its corpus every evaluation, so it gets a sub-corpus: the dev
    # queries' judged products plus random fillers up to dev_corpus_size.
    rng = np.random.default_rng(seed)
    dev_ids = set(dev["product_id"])
    fill = corpus.loc[~corpus["product_id"].isin(dev_ids), "product_id"].to_numpy()
    n_fill = max(0, dev_corpus_size - len(dev_ids))
    dev_corpus = sorted(dev_ids | set(rng.choice(fill, size=min(n_fill, len(fill)), replace=False).tolist()))
    stats["dev_corpus_products"] = len(dev_corpus)

    corpus.to_parquet(out_dir / "corpus.parquet", index=False)
    for name, df in (("train", train), ("dev", dev), ("test", test)):
        df.to_parquet(out_dir / f"{name}_judgements.parquet", index=False)
    # Two small files for the notebook on the laptop, which never holds the whole corpus:
    # a random sample of passages, and the complete judged lists (with text) of a few test queries.
    corpus.sample(n=min(10_000, len(corpus)), random_state=seed).to_parquet(out_dir / "corpus_sample.parquet", index=False)
    sample_q = rng.choice(np.sort(test["query_id"].unique()), size=min(25, test["query_id"].nunique()), replace=False)
    test[test["query_id"].isin(set(sample_q.tolist()))].merge(corpus, on="product_id", how="left").to_parquet(out_dir / "test_examples.parquet", index=False)
    (out_dir / "dev_corpus_ids.json").write_text(json.dumps(dev_corpus))
    stats["build_seconds"] = round(time.time() - t0, 1)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    log(f"benchmark written to {out_dir} in {stats['build_seconds']}s")
    return stats


def load_benchmark(data_dir: Path, corpus_limit: int = 0, max_test_queries: int = 0, max_train_queries: int = 0, seed: int = 42) -> dict:
    """Read the benchmark files. The limits define a *smoke slice*: a random subset of test queries, and an
    evaluation corpus made of those queries' judged products plus random fillers up to corpus_limit. Same seed → same slice in every script, so BM25 and dense rankings can be fused.
    The test queries and the corpus fillers come from one generator, the training queries from a separate one: the evaluation slice
    must not depend on whether a script passed a training cap (BM25 never does, the training CLI may)."""
    d = Path(data_dir)
    corpus = pd.read_parquet(d / "corpus.parquet")
    train, dev, test = (pd.read_parquet(d / f"{n}_judgements.parquet") for n in ("train", "dev", "test"))
    dev_corpus_ids = json.loads((d / "dev_corpus_ids.json").read_text())
    stats = json.loads((d / "stats.json").read_text())
    rng = np.random.default_rng(seed)
    train_rng = np.random.default_rng([seed, 1])
    if max_test_queries:
        keep = rng.choice(np.sort(test["query_id"].unique()), size=min(max_test_queries, test["query_id"].nunique()), replace=False)
        test = test[test["query_id"].isin(set(keep.tolist()))].reset_index(drop=True)
    if max_train_queries:
        keep = train_rng.choice(np.sort(train["query_id"].unique()), size=min(max_train_queries, train["query_id"].nunique()), replace=False)
        train = train[train["query_id"].isin(set(keep.tolist()))].reset_index(drop=True)
    eval_corpus = corpus
    if corpus_limit and corpus_limit < len(corpus):
        must = set(test["product_id"])
        fill = corpus.loc[~corpus["product_id"].isin(must), "product_id"].to_numpy()
        n_fill = max(0, corpus_limit - len(must))
        keep_ids = must | set(rng.choice(fill, size=min(n_fill, len(fill)), replace=False).tolist())
        eval_corpus = corpus[corpus["product_id"].isin(keep_ids)].reset_index(drop=True)
    return {"corpus": corpus, "eval_corpus": eval_corpus, "train": train, "dev": dev, "test": test,
            "dev_corpus_ids": dev_corpus_ids, "stats": stats,
            "slice": {"corpus_limit": corpus_limit, "max_test_queries": max_test_queries, "max_train_queries": max_train_queries,
                      "seed": seed, "eval_corpus_products": int(len(eval_corpus)), "test_queries": int(test["query_id"].nunique()),
                      "train_queries": int(train["query_id"].nunique())}}


def training_examples(data: dict, negatives: str, rng: np.random.Generator, max_pos: int = 3, mined: dict[int, list[str]] | None = None) -> tuple[list[dict], dict]:
    """Training rows for the contrastive loss, (anchor, positive[, negative]): up to `max_pos` Exact products per query, each paired with one
    negative of the requested kind. Queries without a negative of that kind fall back to a random product."""
    text = dict(zip(data["corpus"]["product_id"], data["corpus"]["text"], strict=True))
    all_ids = data["corpus"]["product_id"].to_numpy()
    rows, fallback = [], 0
    for qid, g in data["train"].groupby("query_id"):
        q = g["query"].iloc[0]
        exact = [p for p in g.loc[g["esci_label"] == "E", "product_id"] if p in text]
        if not exact:
            continue
        rng.shuffle(exact)
        if negatives == "labelled":
            pool = [p for p in g.loc[g["esci_label"].isin(["S", "C", "I"]), "product_id"] if p in text]
        elif negatives == "substitutes":
            pool = [p for p in g.loc[g["esci_label"] == "S", "product_id"] if p in text]
        elif negatives == "mined":
            pool = [p for p in (mined or {}).get(int(qid), []) if p in text]
        else:
            pool = []
        exact_set = set(g.loc[g["esci_label"] == "E", "product_id"])
        for p in exact[:max_pos]:
            row = {"anchor": q, "positive": text[p]}
            if negatives != "none":
                if pool:
                    neg = pool[rng.integers(len(pool))]
                else:
                    fallback += 1
                    neg = all_ids[rng.integers(len(all_ids))]
                    while neg in exact_set:
                        neg = all_ids[rng.integers(len(all_ids))]
                row["negative"] = text[neg]
            rows.append(row)
    info = {"rows": len(rows), "queries": int(data["train"]["query_id"].nunique()), "negatives": negatives,
            "random_fallback_rows": fallback, "columns": list(rows[0]) if rows else []}
    return rows, info


def queries_of(df: pd.DataFrame) -> pd.DataFrame:
    """One row per query: query_id, query."""
    return df.drop_duplicates("query_id")[["query_id", "query"]].sort_values("query_id").reset_index(drop=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out-dir", default=None, help="default data/processed/<locale>-<version>")
    p.add_argument("--locale", default="us")
    p.add_argument("--version", default="small", choices=["small", "large"])
    p.add_argument("--dev-queries", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--download", action="store_true", help="fetch the raw files first (products is 1.1 GB)")
    a = p.parse_args()
    if a.download:
        download(a.raw_dir, ("examples", "products", "sources"))
    build(a.raw_dir, a.out_dir or f"data/processed/{a.locale}-{a.version}", a.locale, a.version, a.dev_queries, seed=a.seed)
