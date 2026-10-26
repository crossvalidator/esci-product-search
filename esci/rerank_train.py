"""Stage 2: a cross-encoder reranker over the stage-1 candidates, trained on the same ESCI judgements.

A cross-encoder reads the query and one product *together* through a BERT-style model and outputs a single
relevance score. It is far more accurate than comparing two independent vectors, and far too slow to run over
482k products, so it only ever sees the top-k candidates (k = 20 here) that stage 1 hands it. It is judged on
the order of that short list: recall@3 / hit@3 / MRR@3 / nDCG@3 for the 3 products that go to the LLM, and
nDCG@10 as the standard listwise number. recall@20 cannot change (reranking does not add candidates).

Training rows come from the judgements: (query, product text, label) with label 1 for Exact and 0 for
Substitute / Complement / Irrelevant, pointwise binary cross-entropy. `--loss lambda` instead trains
listwise on the graded gains (E 1.0, S 0.1, C 0.01, I 0) with LambdaLoss.

Examples
--------
python esci/rerank_train.py --eval-only --compare cross-encoder/ms-marco-MiniLM-L-6-v2 BAAI/bge-reranker-base \
    --candidates outputs/embed-v1-substitutes-hybrid/rankings/hybrid(bm25+after).parquet
python esci/rerank_train.py --model BAAI/bge-reranker-base --epochs 1 --candidates <same>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments, losses
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator

import esci

GAINS = esci.metrics.GAINS
CUTOFFS = (3, 10, 20)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="BAAI/bge-reranker-base")
    p.add_argument("--data-dir", default="data/processed/us-small")
    p.add_argument("--output-dir", default="outputs/rerank-v1")
    p.add_argument("--run-name", default="rerank-v1")
    p.add_argument("--candidates", required=True, help="stage-1 rankings parquet (query_id, rank, product_id) to rerank")
    p.add_argument("--k", type=int, default=20, help="how many stage-1 candidates the reranker sees per query")
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--max-length", type=int, default=320, help="query + product tokens")
    p.add_argument("--loss", choices=["bce", "lambda"], default="bce")
    p.add_argument("--max-positives-per-query", type=int, default=4)
    p.add_argument("--max-negatives-per-query", type=int, default=8)
    p.add_argument("--predict-batch-size", type=int, default=256)
    p.add_argument("--max-test-queries", type=int, default=0)
    p.add_argument("--max-train-queries", type=int, default=0)
    p.add_argument("--compare", nargs="*", default=[], help="off-the-shelf rerankers to evaluate alongside")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--fine-tuned", default=None, help="with --eval-only: a saved fine-tuned reranker to evaluate too")
    p.add_argument("--skip-before", action="store_true")
    p.add_argument("--report-to", default="none")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_reranker(path: str, max_length: int, for_training: bool = False) -> CrossEncoder:
    cuda = torch.cuda.is_available()
    kwargs = {} if for_training or not cuda else {"model_kwargs": {"torch_dtype": torch.float16}}
    return CrossEncoder(path, device="cuda" if cuda else "cpu", max_length=max_length, num_labels=1, **kwargs)


def candidates_for(test: pd.DataFrame, rankings_path: Path, k: int) -> dict[int, list[str]]:
    ranked = esci.metrics.load_rankings(rankings_path)
    qids = set(test["query_id"].unique())
    return {q: pids[:k] for q, pids in ranked.items() if q in qids}


def rerank(model: CrossEncoder, cands: dict[int, list[str]], queries: dict[int, str], text: dict[str, str], batch_size: int) -> tuple[dict, dict]:
    pairs, index = [], []
    for q, pids in cands.items():
        for p in pids:
            pairs.append((queries[q], text[p]))
            index.append((q, p))
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    by_q: dict[int, list[tuple[str, float]]] = {}
    for (q, p), s in zip(index, scores):
        by_q.setdefault(q, []).append((p, float(s)))
    ranked = {q: [p for p, _ in sorted(v, key=lambda t: -t[1])] for q, v in by_q.items()}
    sc = {q: [s for _, s in sorted(v, key=lambda t: -t[1])] for q, v in by_q.items()}
    return ranked, sc


def evaluate_reranker(path: str | None, label: str, cands: dict, data: dict, text: dict, out: Path, args: argparse.Namespace) -> dict:
    """path=None scores the stage-1 order itself (the 'no reranker' row)."""
    queries = dict(zip(data["test"]["query_id"], data["test"]["query"]))
    qrels = esci.metrics.qrels_from_frame(data["test"])
    t0 = time.time()
    if path is None:
        ranked, sc, load_s, params = dict(cands), None, 0.0, 0
    else:
        model = load_reranker(path, args.max_length)
        load_s = time.time() - t0
        params = round(sum(p.numel() for p in model.model.parameters()) / 1e6, 1)
        ranked, sc = rerank(model, cands, queries, text, args.predict_batch_size)
        del model
        torch.cuda.empty_cache()
    res = esci.metrics.evaluate(ranked, qrels, ks=CUTOFFS)
    safe = label.replace("/", "__")
    esci.metrics.save_rankings(out / "rankings" / f"{safe}.parquet", ranked, sc)
    res["per_query"].to_parquet(out / "rankings" / f"{safe}.per_query.parquet", index=False)
    m = res["metrics"]
    n_pairs = sum(len(v) for v in cands.values())
    print(f"{label:44s} recall@3 {m['recall@3']:.3f}  hit@3 {m['hit@3']:.3f}  mrr@3 {m['mrr@3']:.3f}  ndcg@3 {m['ndcg@3']:.3f}  "
          f"ndcg@10 {m['ndcg@10']:.3f}  ({n_pairs} pairs, load {load_s:.0f}s, score {time.time() - t0 - load_s:.0f}s)", flush=True)
    return {"model": path, "params_M": params, "k": args.k, "pairs_scored": n_pairs, "load_seconds": round(load_s, 1),
            "score_seconds": round(time.time() - t0 - load_s, 1), "metrics": m, "rankings_file": f"rankings/{safe}.parquet"}


def training_rows(data: dict, text: dict, rng: np.random.Generator, max_pos: int, max_neg: int, listwise: bool) -> tuple[list[dict], dict]:
    """Pointwise: (query, passage, label∈{0,1}). Listwise: (query, [docs], [graded gains]) per query."""
    rows, n_pos, n_neg = [], 0, 0
    for qid, g in data["train"].groupby("query_id"):
        q = g["query"].iloc[0]
        pos = [p for p in g.loc[g["esci_label"] == "E", "product_id"] if p in text]
        neg = g[g["esci_label"] != "E"]
        neg = [(p, l) for p, l in zip(neg["product_id"], neg["esci_label"]) if p in text]
        if not pos or not neg:
            continue
        rng.shuffle(pos)
        rng.shuffle(neg)
        pos, neg = pos[:max_pos], neg[:max_neg]
        n_pos += len(pos)
        n_neg += len(neg)
        if listwise:
            docs = [text[p] for p in pos] + [text[p] for p, _ in neg]
            labels = [GAINS["E"]] * len(pos) + [GAINS[l] for _, l in neg]
            rows.append({"query": q, "docs": docs, "labels": labels})
        else:
            rows.extend({"query": q, "passage": text[p], "label": 1.0} for p in pos)
            rows.extend({"query": q, "passage": text[p], "label": 0.0} for p, _ in neg)
    return rows, {"rows": len(rows), "positives": n_pos, "negatives": n_neg, "queries": int(data["train"]["query_id"].nunique()), "listwise": listwise}


def dev_evaluator(data: dict, text: dict) -> CrossEncoderRerankingEvaluator:
    """Rerank each dev query's own judged list (Exact = positive, rest = negative); cheap, runs every epoch."""
    samples = []
    for qid, g in data["dev"].groupby("query_id"):
        pos = [text[p] for p in g.loc[g["esci_label"] == "E", "product_id"] if p in text]
        neg = [text[p] for p in g.loc[g["esci_label"] != "E", "product_id"] if p in text]
        if pos and neg:
            samples.append({"query": g["query"].iloc[0], "positive": pos, "negative": neg})
    return CrossEncoderRerankingEvaluator(samples, at_k=10, mrr_at_k=3, name="dev", batch_size=128, show_progress_bar=False, write_csv=False)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = esci.data.load_benchmark(args.data_dir, 0, args.max_test_queries, args.max_train_queries, args.seed)
    text = dict(zip(data["corpus"]["product_id"], data["corpus"]["text"]))
    cands = candidates_for(data["test"], Path(args.candidates), args.k)
    missing = [q for q in data["test"]["query_id"].unique() if q not in cands]
    print(f"candidates: {len(cands)} test queries × top-{args.k} from {args.candidates} ({len(missing)} test queries without candidates)", flush=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    report: dict = {"run_name": args.run_name, "model": args.model, "args": vars(args), "slice": data["slice"], "gpu": gpu,
                    "candidates": str(args.candidates), "results": {}}
    t_all = time.time()

    def checkpoint() -> None:
        report["wall_seconds"] = round(time.time() - t_all, 1)
        (out / "report.json").write_text(json.dumps(report, indent=2))

    report["results"]["stage-1 order (no reranker)"] = evaluate_reranker(None, "stage-1 order (no reranker)", cands, data, text, out, args)
    checkpoint()
    if args.eval_only:
        if args.fine_tuned:
            report["results"]["fine-tuned"] = evaluate_reranker(args.fine_tuned, "fine-tuned", cands, data, text, out, args)
            checkpoint()
        for m in args.compare:
            report["results"][m] = evaluate_reranker(m, m, cands, data, text, out, args)
            checkpoint()
        print(f"report written to {out / 'report.json'}")
        return

    if not args.skip_before:
        report["results"]["before"] = evaluate_reranker(args.model, "before", cands, data, text, out, args)
        checkpoint()

    rng = np.random.default_rng(args.seed)
    rows, info = training_rows(data, text, rng, args.max_positives_per_query, args.max_negatives_per_query, args.loss == "lambda")
    report["training_data"] = info
    print(f"training rows: {info}", flush=True)
    train_ds = Dataset.from_list(rows)

    model = load_reranker(args.model, args.max_length, for_training=True)
    if args.loss == "bce":
        pos_weight = torch.tensor(info["negatives"] / max(info["positives"], 1))
        loss = losses.BinaryCrossEntropyLoss(model, pos_weight=pos_weight)
    else:
        loss = losses.LambdaLoss(model)
    targs = CrossEncoderTrainingArguments(
        output_dir=str(out / "checkpoints"), run_name=args.run_name, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, learning_rate=args.lr, warmup_ratio=0.1, lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(), eval_strategy="epoch", save_strategy="no", logging_steps=50, report_to=args.report_to,
        seed=args.seed, dataloader_num_workers=2,
    )
    trainer = CrossEncoderTrainer(model=model, args=targs, train_dataset=train_ds, loss=loss, evaluator=dev_evaluator(data, text))
    t0 = time.time()
    result = trainer.train()
    report["metrics"] = {**result.metrics, "wall_seconds": round(time.time() - t0, 1)}
    report["log_history"] = trainer.state.log_history
    model.save_pretrained(str(out / "model"))
    print(f"fine-tuned reranker saved to {out / 'model'} after {time.time() - t0:.0f}s of training", flush=True)
    del model, trainer
    torch.cuda.empty_cache()
    checkpoint()

    report["results"]["after"] = evaluate_reranker(str(out / "model"), "after", cands, data, text, out, args)
    checkpoint()
    for m in args.compare:
        report["results"][m] = evaluate_reranker(m, m, cands, data, text, out, args)
        checkpoint()
    print(f"report written to {out / 'report.json'}")


if __name__ == "__main__":
    main()
