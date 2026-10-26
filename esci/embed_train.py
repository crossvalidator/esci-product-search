"""Fine-tune or evaluate an embedding model for ESCI product search (stage 1 of a two-stage retriever).

The recipe started life on a toy corpus (18 topics, 54 passages, one right answer per query). What changed for a real benchmark:
- Many relevant products per query (graded judgements), evaluated with `esci.metrics` over a corpus of
  hundreds of thousands of products: recall@10/20/50, hit@k, MRR and graded nDCG, plus saved rankings.
- Negatives come from the data: `labelled` (an S/C/I product judged for the same query), `substitutes`
  (S only, the hardest), `mined` (a top-ranked non-Exact product from the base model), `random`, or `none`.
- A small dev set with its own sub-corpus drives the in-training evaluator; the full test evaluation runs
  once before and once after training, so it does not dominate the GPU bill.
- Smoke-slice flags (`--corpus-limit`, `--max-test-queries`, `--max-train-queries`) for cheap pipeline checks.

Examples
--------
python esci/embed_train.py --data-dir data/processed/us-small --eval-only --compare BAAI/bge-base-en-v1.5
python esci/embed_train.py --data-dir data/processed/us-small --negatives labelled --epochs 1 --batch-size 64
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
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import (
    BatchSamplers,
    SentenceTransformerTrainingArguments,
)

import esci

# BGE v1.5 models were trained with this prefix on short queries (never on passages).
# Qwen3-Embedding takes a one-line task instruction on the query side only; the model card asks for a
# task-specific one, so this is written for shopping queries rather than the generic web-search default.
_BGE = "Represent this sentence for searching relevant passages: "
_QWEN3 = "Instruct: Given a user's shopping query, retrieve the product listings that match it\nQuery: "
QUERY_PROMPTS = {
    "BAAI/bge-base-en-v1.5": _BGE, "BAAI/bge-small-en-v1.5": _BGE, "BAAI/bge-large-en-v1.5": _BGE,
    "Qwen/Qwen3-Embedding-0.6B": _QWEN3, "Qwen/Qwen3-Embedding-4B": _QWEN3, "Qwen/Qwen3-Embedding-8B": _QWEN3,
}
NEGATIVES = ("labelled", "substitutes", "mined", "random", "none")


def load_model(name_or_path: str, max_seq_length: int, for_training: bool = False) -> SentenceTransformer:
    """For evaluation the weights are loaded in half precision (bfloat16 for Qwen3-Embedding, float16 for the BERT-style
    models), which doubles encode throughput on an A10G at no measurable cost in ranking quality. Training keeps
    float32 weights and lets the trainer autocast to bfloat16. Decoder-based embedding models (Qwen3-Embedding) use
    last-token pooling, so the tokenizer must pad on the left; the 8B one needs a 48 GB GPU."""
    cuda = torch.cuda.is_available()
    kwargs = {}
    if "Qwen3-Embedding" in name_or_path:
        kwargs = {"model_kwargs": {"torch_dtype": torch.bfloat16 if cuda else torch.float32},
                  "tokenizer_kwargs": {"padding_side": "left"}}
    elif cuda and not for_training:
        kwargs = {"model_kwargs": {"torch_dtype": torch.float16}}
    model = SentenceTransformer(name_or_path, device="cuda" if cuda else "cpu", **kwargs)
    model.max_seq_length = max_seq_length
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--data-dir", default="data/processed/us-small")
    p.add_argument("--output-dir", default="outputs/embed-v1")
    p.add_argument("--run-name", default="embed-v1")
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--max-seq-length", type=int, default=256)
    p.add_argument("--negatives", choices=NEGATIVES, default="labelled")
    p.add_argument("--max-positives-per-query", type=int, default=3,
                   help="training rows per query (queries have 1-80 Exact products; a cap keeps epochs short and balanced)")
    p.add_argument("--mine-top-k", type=int, default=30, help="with --negatives mined: sample the negative from the base model's top k")
    p.add_argument("--encode-batch-size", type=int, default=256)
    p.add_argument("--top-k", type=int, default=100, help="ranking depth saved per query (hybrid fusion reads it)")
    p.add_argument("--corpus-limit", type=int, default=0, help="smoke slice: evaluation corpus size (0 = all)")
    p.add_argument("--max-test-queries", type=int, default=0, help="smoke slice: number of test queries (0 = all)")
    p.add_argument("--max-train-queries", type=int, default=0, help="smoke slice: number of training queries (0 = all)")
    p.add_argument("--no-query-prompt", action="store_true", help="with --eval-only: embed queries without the model's instruction (ablation)")
    p.add_argument("--compare", nargs="*", default=[], help="off-the-shelf models to evaluate alongside")
    p.add_argument("--eval-only", action="store_true", help="no training: evaluate --compare models (and --fine-tuned)")
    p.add_argument("--fine-tuned", default=None, help="with --eval-only: path of a saved fine-tuned model to evaluate too")
    p.add_argument("--skip-before", action="store_true", help="do not re-evaluate the base model before training (its baseline is already in outputs/)")
    p.add_argument("--report-to", default="none")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def encode(model: SentenceTransformer, texts: list[str], prompt: str | None, batch_size: int, label: str) -> torch.Tensor:
    """Normalised fp16 embeddings on the GPU, encoded in chunks with a progress line per chunk."""
    chunk, parts, t0 = 50_000, [], time.time()
    for i in range(0, len(texts), chunk):
        emb = model.encode(texts[i:i + chunk], prompt=prompt, batch_size=batch_size, normalize_embeddings=True,
                           convert_to_tensor=True, show_progress_bar=False)
        parts.append(emb.half())
        if len(texts) > chunk:
            print(f"  {label}: encoded {min(i + chunk, len(texts))}/{len(texts)} in {time.time() - t0:.0f}s", flush=True)
    return torch.cat(parts)


def search(q_emb: torch.Tensor, c_emb: torch.Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k by cosine (vectors are normalised, so a dot product), 512 queries at a time."""
    k = min(k, c_emb.shape[0])
    idx, sc = [], []
    for i in range(0, q_emb.shape[0], 512):
        s = q_emb[i:i + 512] @ c_emb.T
        top = s.topk(k, dim=1)
        idx.append(top.indices.cpu().numpy())
        sc.append(top.values.float().cpu().numpy())
    return np.concatenate(idx), np.concatenate(sc)


def evaluate_model(path: str, label: str, data: dict, prompt_key: str | None, out_dir: Path, args: argparse.Namespace,
                   keep_corpus_emb: bool = False) -> tuple[dict, torch.Tensor | None]:
    """Encode the evaluation corpus and the test queries, rank, score with esci.metrics, save the rankings."""
    t0 = time.time()
    model = load_model(path, args.max_seq_length)
    prompt = QUERY_PROMPTS.get(prompt_key) if prompt_key else None
    load_s = time.time() - t0
    corpus = data["eval_corpus"]
    t1 = time.time()
    c_emb = encode(model, corpus["text"].tolist(), None, args.encode_batch_size, f"{label} corpus")
    corpus_s = time.time() - t1
    queries = esci.data.queries_of(data["test"])
    t2 = time.time()
    q_emb = encode(model, queries["query"].tolist(), prompt, args.encode_batch_size, f"{label} queries")
    idx, sc = search(q_emb, c_emb, args.top_k)
    query_s = time.time() - t2
    pids = corpus["product_id"].to_numpy()
    qids = queries["query_id"].tolist()
    ranked = {int(q): pids[row].tolist() for q, row in zip(qids, idx)}
    scores = {int(q): row.tolist() for q, row in zip(qids, sc)}
    res = esci.metrics.evaluate(ranked, esci.metrics.qrels_from_frame(data["test"]))
    safe = label.replace("/", "__")
    esci.metrics.save_rankings(out_dir / "rankings" / f"{safe}.parquet", ranked, scores)
    res["per_query"].to_parquet(out_dir / "rankings" / f"{safe}.per_query.parquet", index=False)
    m = res["metrics"]
    print(f"{label:40s} recall@20 {m['recall@20']:.3f}  recall@50 {m['recall@50']:.3f}  hit@20 {m['hit@20']:.3f}  "
          f"ndcg@20 {m['ndcg@20']:.3f}  (load {load_s:.0f}s, corpus {corpus_s:.0f}s, queries+search {query_s:.0f}s)", flush=True)
    result = {"model": path, "query_prompt": prompt, "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
              "embedding_dim": model.get_sentence_embedding_dimension(), "max_seq_length": args.max_seq_length, "dtype": str(model.dtype),
              "load_seconds": round(load_s, 1), "encode_corpus_seconds": round(corpus_s, 1), "queries_and_search_seconds": round(query_s, 1),
              "corpus_products": int(len(corpus)), "metrics": m, "rankings_file": f"rankings/{safe}.parquet"}
    del model, q_emb
    if not keep_corpus_emb:
        del c_emb
        c_emb = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, c_emb


def mine_negatives(model_path: str, data: dict, c_emb: torch.Tensor | None, args: argparse.Namespace) -> dict[int, list[str]]:
    """For each training query, the base model's top-k products that are not labelled Exact for it. Mined
    negatives may include unjudged products that are in fact relevant (false negatives); that is the known
    trade-off versus the labelled S/C/I negatives, and one of the things the experiments compare."""
    model = load_model(model_path, args.max_seq_length)
    prompt = QUERY_PROMPTS.get(args.model)
    corpus = data["eval_corpus"]
    if c_emb is None:
        c_emb = encode(model, corpus["text"].tolist(), None, args.encode_batch_size, "mining corpus")
    queries = esci.data.queries_of(data["train"])
    q_emb = encode(model, queries["query"].tolist(), prompt, args.encode_batch_size, "mining queries")
    idx, _ = search(q_emb, c_emb, args.mine_top_k)
    pids = corpus["product_id"].to_numpy()
    exact = data["train"][data["train"]["esci_label"] == "E"].groupby("query_id")["product_id"].agg(set).to_dict()
    mined = {int(q): [p for p in pids[row].tolist() if p not in exact.get(int(q), set())] for q, row in zip(queries["query_id"], idx)}
    del model, q_emb, c_emb
    torch.cuda.empty_cache()
    return mined


def dev_evaluator(data: dict, prompt: str | None) -> InformationRetrievalEvaluator:
    """In-training check on the dev queries over their small sub-corpus. Cheap enough to run every epoch."""
    text = dict(zip(data["corpus"]["product_id"], data["corpus"]["text"]))
    dev = data["dev"]
    queries = esci.data.queries_of(dev)
    relevant = dev[dev["esci_label"] == "E"].groupby("query_id")["product_id"].agg(set).to_dict()
    return InformationRetrievalEvaluator(
        queries={str(q): t for q, t in zip(queries["query_id"], queries["query"]) if q in relevant},
        corpus={p: text[p] for p in data["dev_corpus_ids"] if p in text},
        relevant_docs={str(q): set(v) for q, v in relevant.items()},
        accuracy_at_k=[20], precision_recall_at_k=[10, 20, 50], mrr_at_k=[20], ndcg_at_k=[20], map_at_k=[20],
        query_prompt=prompt, name="dev", batch_size=128, corpus_chunk_size=50_000, write_csv=False, show_progress_bar=False,
    )


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = esci.data.load_benchmark(args.data_dir, args.corpus_limit, args.max_test_queries, args.max_train_queries, args.seed)
    print(f"benchmark slice: {data['slice']}", flush=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    report: dict = {"run_name": args.run_name, "model": args.model, "args": vars(args), "slice": data["slice"], "gpu": gpu, "results": {}}
    t_all = time.time()

    def checkpoint() -> None:
        # written after every model so that a crash or timeout on a later model keeps the earlier results
        report["wall_seconds"] = round(time.time() - t_all, 1)
        (out / "report.json").write_text(json.dumps(report, indent=2))

    if args.eval_only:
        if args.fine_tuned:
            report["results"]["fine-tuned"], _ = evaluate_model(args.fine_tuned, "fine-tuned", data, args.model, out, args)
            checkpoint()
        for m in args.compare:
            report["results"][m], _ = evaluate_model(m, m, data, None if args.no_query_prompt else m, out, args)
            checkpoint()
        print(f"report written to {out / 'report.json'}")
        return

    c_emb = None
    if not args.skip_before or args.negatives == "mined":
        report["results"]["before"], c_emb = evaluate_model(args.model, "before", data, args.model, out, args, keep_corpus_emb=args.negatives == "mined")
        checkpoint()
    mined = mine_negatives(args.model, data, c_emb, args) if args.negatives == "mined" else None
    del c_emb
    rng = np.random.default_rng(args.seed)
    rows, info = esci.data.training_examples(data, args.negatives, rng, args.max_positives_per_query, mined)
    train_ds = Dataset.from_list(rows)
    report["training_data"] = info
    print(f"training rows: {info}", flush=True)

    prompt = QUERY_PROMPTS.get(args.model)
    model = load_model(args.model, args.max_seq_length, for_training=True)
    loss = losses.MultipleNegativesRankingLoss(model)
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(out / "checkpoints"),
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        batch_sampler=BatchSamplers.NO_DUPLICATES,   # the same query twice in a batch would make its other Exact product a false negative
        prompts={"anchor": prompt} if prompt else None,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to=args.report_to,
        seed=args.seed,
        dataloader_num_workers=2,
    )
    trainer = SentenceTransformerTrainer(model=model, args=targs, train_dataset=train_ds, loss=loss, evaluator=dev_evaluator(data, prompt))
    t0 = time.time()
    result = trainer.train()
    report["metrics"] = {**result.metrics, "wall_seconds": round(time.time() - t0, 1)}
    report["log_history"] = trainer.state.log_history
    model.save_pretrained(str(out / "model"))
    print(f"fine-tuned model saved to {out / 'model'} after {time.time() - t0:.0f}s of training", flush=True)
    del model, trainer
    torch.cuda.empty_cache()

    checkpoint()
    report["results"]["after"], _ = evaluate_model(str(out / "model"), "after", data, args.model, out, args)
    checkpoint()
    for m in args.compare:
        report["results"][m], _ = evaluate_model(m, m, data, m, out, args)
        checkpoint()
    print(f"report written to {out / 'report.json'}")


if __name__ == "__main__":
    main()
