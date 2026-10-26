"""Lexical and hybrid baselines for ESCI product search, on CPU.

- BM25 (Best Match 25): the classic term-weighting ranking function; no model, no training. It is strong
  wherever the query contains a brand, model number or exact product word, which is most shopping queries.
- Hybrid: Reciprocal Rank Fusion of the BM25 ranking with each dense ranking passed in (saved by
  esci/embed_train.py), the standard production pattern: both retrievers propose candidates, the union is reranked.

    python -m esci.baselines --data-dir data/processed/us-small --output-dir outputs/baselines \
        --dense-rankings outputs/baselines/rankings/BAAI__bge-base-en-v1.5.parquet
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import bm25s
import Stemmer

import esci


def bm25_rankings(corpus_texts: list[str], corpus_ids, queries: list[tuple[int, str]], k: int) -> tuple[dict, dict, dict]:
    t0 = time.time()
    stemmer = Stemmer.Stemmer("english")
    tokens = bm25s.tokenize(corpus_texts, stopwords="en", stemmer=stemmer, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=False)
    index_s = time.time() - t0
    t1 = time.time()
    q_tokens = bm25s.tokenize([q for _, q in queries], stopwords="en", stemmer=stemmer, show_progress=False)
    k = min(k, len(corpus_texts))
    idx, sc = retriever.retrieve(q_tokens, k=k, show_progress=False, n_threads=-1)
    search_s = time.time() - t1
    ranked, scores = {}, {}
    for (qid, _), row, srow in zip(queries, idx, sc, strict=True):
        ranked[int(qid)], scores[int(qid)] = stable_order([corpus_ids[j] for j in row], srow.tolist())
    return ranked, scores, {"index_seconds": round(index_s, 1), "search_seconds": round(search_s, 1)}


def stable_order(pids: list[str], scores: list[float], decimals: int = 6) -> tuple[list[str], list[float]]:
    """Products with equal BM25 scores (identical titles are common in a catalogue) came back in an order that differed
    between two otherwise identical runs: floating-point sums in the scoring differ by an ulp and flip the tie. recall@20 was
    unaffected, MRR and nDCG moved in the third decimal. Ordering by the score rounded to `decimals`, then by product id,
    makes the ranking a function of the data alone."""
    order = sorted(range(len(pids)), key=lambda i: (-round(scores[i], decimals), pids[i]))
    return [pids[i] for i in order], [scores[i] for i in order]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/processed/us-small")
    p.add_argument("--output-dir", default="outputs/baselines")
    p.add_argument("--run-name", default="baselines")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--corpus-limit", type=int, default=0)
    p.add_argument("--max-test-queries", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dense-rankings", nargs="*", default=[], help="rankings parquet files to fuse with BM25 (label = file stem)")
    p.add_argument("--rrf-k", type=int, default=60)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = esci.data.load_benchmark(args.data_dir, args.corpus_limit, args.max_test_queries, seed=args.seed)
    print(f"benchmark slice: {data['slice']}", flush=True)
    corpus = data["eval_corpus"]
    queries = esci.data.queries_of(data["test"])
    qrels = esci.metrics.qrels_from_frame(data["test"])
    report: dict = {"run_name": args.run_name, "slice": data["slice"], "versions": esci.versions("bm25s", "PyStemmer", "pandas"), "results": {}}
    t_all = time.time()

    ranked, scores, timing = bm25_rankings(corpus["text"].tolist(), corpus["product_id"].tolist(),
                                           list(zip(queries["query_id"], queries["query"], strict=True)), args.top_k)
    res = esci.metrics.evaluate(ranked, qrels)
    esci.metrics.save_rankings(out / "rankings" / "bm25.parquet", ranked, scores)
    res["per_query"].to_parquet(out / "rankings" / "bm25.per_query.parquet", index=False)
    report["results"]["bm25"] = {"model": "BM25 (bm25s, English stopwords + Snowball stemmer)", "params_M": 0, "metrics": res["metrics"],
                                 "corpus_products": int(len(corpus)), "rankings_file": "rankings/bm25.parquet", **timing}
    m = res["metrics"]
    print(f"{'bm25':40s} recall@20 {m['recall@20']:.3f}  recall@50 {m['recall@50']:.3f}  hit@20 {m['hit@20']:.3f}  "
          f"ndcg@20 {m['ndcg@20']:.3f}  (index {timing['index_seconds']}s, search {timing['search_seconds']}s)", flush=True)

    for f in args.dense_rankings:
        dense = esci.metrics.load_rankings(f)
        label = f"hybrid(bm25+{Path(f).stem})"
        fused = esci.metrics.rrf([ranked, dense], k=args.rrf_k, depth=args.top_k)
        res = esci.metrics.evaluate(fused, qrels)
        safe = label.replace("/", "__")
        esci.metrics.save_rankings(out / "rankings" / f"{safe}.parquet", fused)
        res["per_query"].to_parquet(out / "rankings" / f"{safe}.per_query.parquet", index=False)
        report["results"][label] = {"model": f"RRF(k={args.rrf_k}) of BM25 and {Path(f).stem}", "params_M": None, "metrics": res["metrics"],
                                    "corpus_products": int(len(corpus)), "rankings_file": f"rankings/{safe}.parquet", "dense_rankings": str(f)}
        m = res["metrics"]
        print(f"{label:40s} recall@20 {m['recall@20']:.3f}  recall@50 {m['recall@50']:.3f}  hit@20 {m['hit@20']:.3f}  ndcg@20 {m['ndcg@20']:.3f}", flush=True)

    report["wall_seconds"] = round(time.time() - t_all, 1)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"report written to {out / 'report.json'}")


if __name__ == "__main__":
    main()
