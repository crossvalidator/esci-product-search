"""Every heavy step of the ESCI project on Modal: data download and slicing (CPU), dense evaluation and
fine-tuning (GPU), BM25 and hybrid (CPU). The laptop only launches jobs and downloads reports.

    modal run jobs/modal_esci.py::data                                   # download 1.1 GB on Modal, build the us-small benchmark
    modal run jobs/modal_esci.py::baselines --run-name baselines-smoke --corpus-limit 20000 --max-test-queries 500
    modal run jobs/modal_esci.py::baselines --run-name baselines-us-small --models BAAI/bge-base-en-v1.5,BAAI/bge-large-en-v1.5
    modal run jobs/modal_esci.py::compare --run-name compare-qwen --models Qwen/Qwen3-Embedding-0.6B,Qwen/Qwen3-Embedding-8B --gpu L40S
    modal run jobs/modal_esci.py::train --run-name embed-v1 --negatives labelled --epochs 1

Volumes: `esci-data` (raw files + processed benchmark), `esci-runs` (reports, rankings, models),
`hf-cache` (Hugging Face model weights: downloaded once, reused by every run).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import modal

from jobs.common import RATES, est_cost  # noqa: F401  (re-exported: the notebooks call modal_esci.est_cost)

app = modal.App("esci-product-search")

# The image is installed from jobs/requirements-gpu.txt: every package frozen (torch 2.14.1, sentence-transformers 6.2.0,
# transformers 5.20.2, datasets 5.1.0, accelerate 1.15.0, bm25s 0.3.15 and all their dependencies), read back with `pip freeze`
# from the image the results in FINDINGS.md were produced with. Every report.json records the versions it ran with. Bump
# deliberately: edit the file, rebuild, re-run the smoke slices before trusting the tables again.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(Path(__file__).with_name("requirements-gpu.txt")))
    .add_local_python_source("esci", "jobs")  # the library package and this one, mounted at /root; the container imports both
)
data_vol = modal.Volume.from_name("esci-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("esci-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
DATA, RUNS, HF = "/vol/data", "/vol/runs", "/root/.cache/huggingface"
VOLUMES = {DATA: data_vol, RUNS: runs_vol, HF: hf_cache}

def slice_dir(locale: str, version: str) -> str:
    return f"{DATA}/processed/{locale}-{version}"


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd="/root")


def _finish(out: Path, t0: float) -> dict:
    """Add the container wall time (the billed quantity) to the saved report, commit the Volume, return it."""
    report = json.loads((out / "report.json").read_text())
    report["container_seconds"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    runs_vol.commit()
    return report


@app.function(image=image, cpu=4, memory=16_384, timeout=3 * 3600, volumes={DATA: data_vol})
def build_data(locale: str = "us", version: str = "small", dev_queries: int = 500, seed: int = 42) -> dict:
    """Download the raw ESCI files into the esci-data Volume (skipped if present) and build the benchmark."""
    import esci

    t0 = time.time()
    esci.data.download(Path(DATA) / "raw", ("examples", "products", "sources"))
    data_vol.commit()
    stats = esci.data.build(Path(DATA) / "raw", slice_dir(locale, version), locale, version, dev_queries, seed=seed)
    data_vol.commit()
    stats["wall_seconds"] = round(time.time() - t0, 1)
    return stats


@app.function(image=image, gpu="A10G", volumes=VOLUMES, timeout=4 * 3600)
def evaluate_dense(run_name: str, models: list[str], locale: str = "us", version: str = "small", fine_tuned_run: str | None = None,
                   query_prompt: bool = True, corpus_limit: int = 0, max_test_queries: int = 0, max_seq_length: int = 256,
                   encode_batch_size: int = 256, seed: int = 42) -> dict:
    """No training: encode the corpus and test queries with each model, rank, score, save rankings."""
    t0 = time.time()
    out = Path(RUNS) / run_name
    cmd = [sys.executable, "-m", "esci.embed_train", "--eval-only", "--data-dir", slice_dir(locale, version), "--output-dir", str(out),
           "--run-name", run_name, "--corpus-limit", str(corpus_limit), "--max-test-queries", str(max_test_queries),
           "--max-seq-length", str(max_seq_length), "--encode-batch-size", str(encode_batch_size), "--seed", str(seed), "--compare", *models]
    if fine_tuned_run:
        cmd += ["--fine-tuned", str(Path(RUNS) / fine_tuned_run / "model")]
    if not query_prompt:
        cmd.append("--no-query-prompt")
    _run(cmd)
    return _finish(out, t0)


@app.function(image=image, cpu=8, memory=32_768, volumes=VOLUMES, timeout=2 * 3600)
def bm25_and_hybrid(run_name: str, locale: str = "us", version: str = "small", dense_rankings: list[str] = (),
                    corpus_limit: int = 0, max_test_queries: int = 0, seed: int = 42) -> dict:
    """BM25 over the same slice, then Reciprocal Rank Fusion with each dense ranking ("<run>/rankings/<file>.parquet")."""
    t0 = time.time()
    out = Path(RUNS) / run_name
    runs_vol.reload()
    cmd = [sys.executable, "-m", "esci.baselines", "--data-dir", slice_dir(locale, version), "--output-dir", str(out), "--run-name", run_name,
           "--corpus-limit", str(corpus_limit), "--max-test-queries", str(max_test_queries), "--seed", str(seed),
           "--dense-rankings", *[str(Path(RUNS) / r) for r in dense_rankings]]
    _run(cmd)
    return _finish(out, t0)


@app.function(image=image, gpu="A10G", volumes=VOLUMES, timeout=6 * 3600)
def train_and_evaluate(run_name: str, model: str = "BAAI/bge-base-en-v1.5", locale: str = "us", version: str = "small",
                       epochs: float = 1, lr: float = 2e-5, batch_size: int = 64, negatives: str = "labelled",
                       max_positives_per_query: int = 3, compare: list[str] = (), max_steps: int = -1, max_seq_length: int = 256,
                       corpus_limit: int = 0, max_test_queries: int = 0, max_train_queries: int = 0, seed: int = 42,
                       skip_before: bool = False) -> dict:
    """Evaluate before, fine-tune, evaluate after (and any --compare models) on the GPU; everything stays in esci-runs."""
    t0 = time.time()
    out = Path(RUNS) / run_name
    cmd = [sys.executable, "-m", "esci.embed_train", "--model", model, "--data-dir", slice_dir(locale, version), "--output-dir", str(out),
           "--run-name", run_name, "--epochs", str(epochs), "--lr", str(lr), "--batch-size", str(batch_size), "--negatives", negatives,
           "--max-positives-per-query", str(max_positives_per_query), "--max-steps", str(max_steps), "--max-seq-length", str(max_seq_length),
           "--corpus-limit", str(corpus_limit), "--max-test-queries", str(max_test_queries), "--max-train-queries", str(max_train_queries),
           "--seed", str(seed), "--compare", *compare] + (["--skip-before"] if skip_before else [])
    _run(cmd)
    return _finish(out, t0)


@app.function(image=image, gpu="A10G", volumes=VOLUMES, timeout=4 * 3600)
def rerank_train_and_evaluate(run_name: str, candidates: str, model: str = "BAAI/bge-reranker-base", locale: str = "us", version: str = "small",
                              k: int = 20, epochs: float = 1, lr: float = 2e-5, batch_size: int = 32, loss: str = "bce",
                              max_positives_per_query: int = 4, max_negatives_per_query: int = 8, compare: list[str] = (),
                              eval_only: bool = False, fine_tuned_run: str | None = None, skip_before: bool = False, max_steps: int = -1,
                              max_length: int = 320, max_test_queries: int = 0, max_train_queries: int = 0, seed: int = 42) -> dict:
    """Stage 2: rerank the top-k of a saved stage-1 ranking ("<run>/rankings/<file>.parquet") with cross-encoders;
    optionally fine-tune one on the training judgements first."""
    t0 = time.time()
    out = Path(RUNS) / run_name
    runs_vol.reload()
    cmd = [sys.executable, "-m", "esci.rerank_train", "--model", model, "--data-dir", slice_dir(locale, version), "--output-dir", str(out),
           "--run-name", run_name, "--candidates", str(Path(RUNS) / candidates), "--k", str(k), "--epochs", str(epochs), "--lr", str(lr),
           "--batch-size", str(batch_size), "--loss", loss, "--max-positives-per-query", str(max_positives_per_query),
           "--max-negatives-per-query", str(max_negatives_per_query), "--max-steps", str(max_steps), "--max-length", str(max_length),
           "--max-test-queries", str(max_test_queries), "--max-train-queries", str(max_train_queries), "--seed", str(seed), "--compare", *compare]
    if eval_only:
        cmd.append("--eval-only")
    if fine_tuned_run:
        cmd += ["--fine-tuned", str(Path(RUNS) / fine_tuned_run / "model")]
    if skip_before:
        cmd.append("--skip-before")
    _run(cmd)
    return _finish(out, t0)


# ---------------------------------------------------------------- downloads (laptop side)

def _fetch(vol: modal.Volume, remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    with open(local, "wb") as fh:
        for chunk in vol.read_file(remote):
            fh.write(chunk)


def download_benchmark(locale: str = "us", version: str = "small", local_root: str = "data/processed", with_corpus: bool = False) -> Path:
    """Bring back the small benchmark files (judgements, stats, dev corpus ids; ~20 MB). The 480k-product corpus
    stays on the Volume unless with_corpus=True."""
    dest = Path(local_root) / f"{locale}-{version}"
    names = ["stats.json", "train_judgements.parquet", "dev_judgements.parquet", "test_judgements.parquet", "dev_corpus_ids.json",
             "corpus_sample.parquet", "test_examples.parquet"]
    if with_corpus:
        names.append("corpus.parquet")
    for n in names:
        _fetch(data_vol, f"processed/{locale}-{version}/{n}", dest / n)
    return dest


def download_run(run_name: str, local_root: str = "outputs", with_model: bool = False) -> Path:
    """report.json and the rankings/ folder (top-100 per test query per model, a few MB each); the model only on request."""
    dest = Path(local_root) / run_name
    for entry in runs_vol.listdir(run_name, recursive=True):
        rel = entry.path.split(f"{run_name}/", 1)[1]
        if entry.type != modal.volume.FileEntryType.FILE or rel.startswith("checkpoints/"):
            continue
        if rel.startswith("model/") and not with_model:
            continue
        _fetch(runs_vol, entry.path, dest / rel)
    return dest


def _print_results(report: dict, gpu: str | None, cpu: float = 0, mem: float = 0) -> None:
    for label, r in report["results"].items():
        m = r["metrics"]
        p = f"{r['params_M']:7.1f}M" if r.get("params_M") else "       -"
        print(f"{label:40s} params {p}  recall@10 {m['recall@10']:.3f}  recall@20 {m['recall@20']:.3f}  recall@50 {m['recall@50']:.3f}  "
              f"hit@20 {m['hit@20']:.3f}  ndcg@20 {m['ndcg@20']:.3f}")
    secs = report.get("container_seconds", report.get("wall_seconds", 0))
    print(f"container time {secs:.0f}s on {gpu or f'{cpu:g} CPU cores'}; estimated cost at list price ${est_cost(secs, gpu, cpu, mem):.3f}")


# ---------------------------------------------------------------- entrypoints

@app.local_entrypoint()
def data(locale: str = "us", version: str = "small", dev_queries: int = 500, seed: int = 42, with_corpus: bool = False):
    """Download the raw files to the esci-data Volume, build the benchmark, bring back the small files."""
    t0 = time.time()
    stats = build_data.remote(locale=locale, version=version, dev_queries=dev_queries, seed=seed)
    dest = download_benchmark(locale, version, with_corpus=with_corpus)
    print(json.dumps({k: v for k, v in stats.items() if k in ("train", "dev", "test", "corpus", "train_judgements_removed_product_overlap")}, indent=2))
    print(f"benchmark files in {dest}/; container {stats['wall_seconds']:.0f}s, estimated cost ${est_cost(stats['wall_seconds'], None, 4, 16):.3f}; "
          f"round trip {(time.time() - t0) / 60:.1f} min")


@app.local_entrypoint()
def baselines(run_name: str = "baselines-smoke", models: str = "BAAI/bge-base-en-v1.5", gpu: str = "A10G", locale: str = "us",
              version: str = "small", corpus_limit: int = 20_000, max_test_queries: int = 500, seed: int = 42, max_seq_length: int = 256):
    """Dense models on the GPU, then BM25 and hybrid (BM25 + each dense model) on CPU, all on one slice.
    Defaults are the smoke slice; pass --corpus-limit 0 --max-test-queries 0 for the full benchmark."""
    model_list = [m for m in models.split(",") if m]
    t0 = time.time()
    dense = evaluate_dense.with_options(gpu=gpu).remote(run_name=run_name, models=model_list, locale=locale, version=version, corpus_limit=corpus_limit,
                                                        max_test_queries=max_test_queries, max_seq_length=max_seq_length, seed=seed)
    _print_results(dense, gpu)
    files = [f"{run_name}/{r['rankings_file']}" for r in dense["results"].values()]
    lex = bm25_and_hybrid.remote(run_name=f"{run_name}-bm25", locale=locale, version=version, dense_rankings=files, corpus_limit=corpus_limit,
                                 max_test_queries=max_test_queries, seed=seed)
    _print_results(lex, None, 8, 32)
    download_run(run_name)
    download_run(f"{run_name}-bm25")
    print(f"reports in outputs/{run_name}/ and outputs/{run_name}-bm25/; round trip {(time.time() - t0) / 60:.1f} min")


@app.local_entrypoint()
def compare(run_name: str = "compare-qwen", models: str = "Qwen/Qwen3-Embedding-0.6B,Qwen/Qwen3-Embedding-8B", gpu: str = "L40S",
            fine_tuned_run: str = "", query_prompt: bool = True, locale: str = "us", version: str = "small",
            corpus_limit: int = 0, max_test_queries: int = 0, seed: int = 42, encode_batch_size: int = 64):
    """Bigger off-the-shelf models (and optionally a fine-tuned run) on an L40S; no training."""
    t0 = time.time()
    report = evaluate_dense.with_options(gpu=gpu).remote(run_name=run_name, models=[m for m in models.split(",") if m], locale=locale, version=version,
                                                         fine_tuned_run=fine_tuned_run or None, query_prompt=query_prompt, corpus_limit=corpus_limit,
                                                         max_test_queries=max_test_queries, encode_batch_size=encode_batch_size, seed=seed)
    _print_results(report, gpu)
    download_run(run_name)
    print(f"report in outputs/{run_name}/; round trip {(time.time() - t0) / 60:.1f} min")


@app.local_entrypoint()
def rerank(run_name: str = "rerank-smoke", candidates: str = "embed-v1-substitutes-hybrid/rankings/hybrid(bm25+after).parquet",
           model: str = "BAAI/bge-reranker-base", gpu: str = "A10G", locale: str = "us", version: str = "small", k: int = 20, epochs: float = 1,
           lr: float = 2e-5, batch_size: int = 32, loss: str = "bce", max_positives_per_query: int = 4, max_negatives_per_query: int = 8,
           compare: str = "", eval_only: bool = False, fine_tuned_run: str = "", skip_before: bool = False, max_steps: int = -1,
           max_length: int = 320, max_test_queries: int = 0, max_train_queries: int = 0, seed: int = 42, download_model: bool = False):
    """Stage 2 reranker; smoke check with --max-steps 30 --max-test-queries 500 --max-train-queries 2000."""
    t0 = time.time()
    report = rerank_train_and_evaluate.with_options(gpu=gpu).remote(
        run_name=run_name, candidates=candidates, model=model, locale=locale, version=version, k=k, epochs=epochs, lr=lr, batch_size=batch_size,
        loss=loss, max_positives_per_query=max_positives_per_query, max_negatives_per_query=max_negatives_per_query,
        compare=[m for m in compare.split(",") if m], eval_only=eval_only, fine_tuned_run=fine_tuned_run or None, skip_before=skip_before,
        max_steps=max_steps, max_length=max_length, max_test_queries=max_test_queries, max_train_queries=max_train_queries, seed=seed)
    for label, r in report["results"].items():
        m = r["metrics"]
        print(f"{label:44s} recall@3 {m['recall@3']:.3f}  hit@3 {m['hit@3']:.3f}  mrr@3 {m['mrr@3']:.3f}  ndcg@3 {m['ndcg@3']:.3f}  ndcg@10 {m['ndcg@10']:.3f}")
    secs = report["container_seconds"]
    print(f"container time {secs:.0f}s on {gpu}; estimated cost at list price ${est_cost(secs, gpu):.3f}")
    download_run(run_name, with_model=download_model)
    print(f"report in outputs/{run_name}/; round trip {(time.time() - t0) / 60:.1f} min")


@app.local_entrypoint()
def train(run_name: str = "embed-v1", model: str = "BAAI/bge-base-en-v1.5", gpu: str = "A10G", epochs: float = 1, lr: float = 2e-5,
          batch_size: int = 64, negatives: str = "labelled", max_positives_per_query: int = 3, compare: str = "", max_steps: int = -1,
          locale: str = "us", version: str = "small", corpus_limit: int = 0, max_test_queries: int = 0, max_train_queries: int = 0,
          seed: int = 42, download_model: bool = False, skip_before: bool = False):
    """Fine-tune on the GPU; smoke check with --max-steps 20 --corpus-limit 20000 --max-test-queries 500 --max-train-queries 2000."""
    t0 = time.time()
    report = train_and_evaluate.with_options(gpu=gpu).remote(
        run_name=run_name, model=model, locale=locale, version=version, epochs=epochs, lr=lr, batch_size=batch_size, negatives=negatives,
        max_positives_per_query=max_positives_per_query, compare=[m for m in compare.split(",") if m], max_steps=max_steps,
        corpus_limit=corpus_limit, max_test_queries=max_test_queries, max_train_queries=max_train_queries, seed=seed, skip_before=skip_before)
    _print_results(report, gpu)
    download_run(run_name, with_model=download_model)
    print(f"report in outputs/{run_name}/; training {report['metrics']['wall_seconds']:.0f}s; round trip {(time.time() - t0) / 60:.1f} min")
