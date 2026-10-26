# ESCI product search: fine-tuning an embedding model for a real retrieval pipeline

A showcase project on the [Amazon Shopping Queries Dataset (ESCI)](https://github.com/amazon-science/esci-data):
real shopping queries, human-judged products (Exact / Substitute / Complement / Irrelevant), Apache-2.0.
Stage 1 of a two-stage retriever is a small embedding model; it is judged on **recall@20**, the share of a
query's Exact products inside the 20 candidates handed to the reranker. Everything heavy runs on Modal;
the laptop reads a 51 MB judgements file, launches jobs and reads reports back. No model weights are loaded locally.

The pattern throughout: data script → training CLI → Modal launcher → `report.json` → executed notebook → `FINDINGS.md`.
Every number in the notebooks and in `FINDINGS.md` comes from an executed cell or a saved report; nothing is typed in.

## Layout

```
esci-product-search/
  01-data-and-baselines.ipynb … 04-spark-delta-lakehouse.ipynb   the executed notebooks (the entry point)
  README.md  FINDINGS.md  SPARK.md                                this file; every result; the Spark reasoning
  esci/        the library: data.py, metrics.py, bootstrap.py, embed_train.py, rerank_train.py, baselines.py, spark.py
  jobs/        the Modal launchers: modal_esci.py (GPU + CPU jobs), modal_spark.py (Spark + Delta Lake)
  notebooks/   build_01.py … build_04.py (write the notebooks) and run_notebook.py (execute cell by cell with a log)
  data/  outputs/  logs/                                           downloaded data, reports brought back from Modal, logs; gitignored
```

Everything is run from the project root (`uv run python notebooks/build_01.py`, `uv run modal run jobs/modal_esci.py::data`);
the notebooks import `esci` and `jobs` from there.

## Setup

```bash
cd esci-product-search
uv sync                                                              # ~370 MB; no torch locally
uv run python -m ipykernel install --user --name esci-product-search --display-name "ESCI product search"
uv run modal setup                                                   # once; logs this machine into your Modal account
```

## Run

```bash
uv run modal run jobs/modal_esci.py::data                                 # downloads 1.1 GB on Modal, builds data/processed/us-small (CPU, cents)
uv run modal run jobs/modal_esci.py::baselines                            # smoke slice: bge-base + BM25 + hybrid on 20k products / 500 queries
uv run modal run jobs/modal_esci.py::baselines --run-name baselines-us-small --corpus-limit 0 --max-test-queries 0 \
    --models BAAI/bge-base-en-v1.5,BAAI/bge-large-en-v1.5           # full benchmark
uv run modal run jobs/modal_esci.py::compare --gpu L40S                   # Qwen3-Embedding-0.6B / -8B, full corpus
uv run modal run jobs/modal_esci.py::train --run-name embed-v1 --negatives labelled --epochs 1   # notebook 02
uv run modal run jobs/modal_esci.py::rerank --run-name rerank-v1 --batch-size 64                 # notebook 03: fine-tune bge-reranker-base over the hybrid top-20
uv run modal run jobs/modal_spark.py::pipeline --run-name spark-all --locales us,es,jp         # notebook 04: Spark + Delta Lake data pipeline, all locales (CPU, cents)
uv run python notebooks/build_01.py && uv run python notebooks/run_notebook.py 01-data-and-baselines.ipynb   # rebuild + execute cell by cell, log in logs/
```

Every Modal entrypoint prints the container time and an estimated cost at list price; the smoke slice
(`--corpus-limit 20000 --max-test-queries 500`) is the default for `baselines` so nothing big runs by accident.

## Reproducing this on your own Modal account

Nothing in the repository is tied to the account it was built on. What you need:

- **A [Modal](https://modal.com) account.** `uv run modal setup` opens the browser once and writes a token to `~/.modal.toml`;
  from then on every `modal run` in this project bills your workspace. At the time of writing the free tier includes
  monthly compute credits that cover this whole project. No other credentials: the ESCI files download from Amazon's public
  GitHub repository and every model (`bge-base`, `bge-large`, `Qwen3-Embedding-0.6B/8B`, `bge-reranker-base`,
  `ms-marco-MiniLM`) is public on the Hugging Face Hub, so no Hugging Face token is needed.
- **The three Modal Volumes are created on first use** (`esci-data`, `esci-runs`, `hf-cache`), in your workspace, by the
  launchers themselves. The images (one with torch and sentence-transformers, one with Java, Spark and Delta Lake) build
  automatically on the first run, in a few minutes.
- **uv and Python 3.12** on the laptop. No GPU, no torch, no Java locally: the laptop only launches jobs and reads reports.

Two ways to run it, from the project root:

1. **Execute the notebooks in order.** Each Modal cell checks `outputs/<run>/report.json`; if the report is there it reads
   it, otherwise it launches the job, waits, and downloads the report. On a fresh clone `outputs/` is empty, so running
   `notebooks/run_notebook.py` on notebooks 01 to 04 in order reproduces every run and leaves the reports behind.
   Notebook 01 also builds the benchmark on Modal and downloads the small files it needs (`data/processed/us-small/`).
2. **The CLI entrypoints** under *Run* above, one job at a time, then execute the notebooks, which will find the reports.

What it costs at Modal list prices (container seconds × rate; the measured numbers are in `FINDINGS.md`):

| Notebook | GPU | Wall time | Cost | Of which |
|---|---|---|---|---|
| 01 data + baselines | A10G, and an L40S for Qwen3-Embedding-8B | about 3 h, mostly the 8B encode | about $6.40 | $5.60 is the one-time Qwen3-8B pass; set `RUN_FULL_QWEN = False` in notebook 01 to skip it |
| 02 fine-tuning (five runs + hybrid + two extra seeds) | A10G | about 1.5 h, runs in parallel | about $2.10 | |
| 03 reranker | A10G | about 25 min | about $0.41 | |
| 04 Spark + Delta Lake | CPU only | about 12 min | about $0.11 | |

About $9 in total. Every launcher prints the container time and the estimated cost when it finishes, and the smoke
slice is the default for `baselines` so nothing big runs by accident.

What to expect from the numbers: evaluation of off-the-shelf models and the Spark pipeline reproduce the tables here
exactly (same data, same weights, fp16 inference). The fine-tuning runs are seeded, but GPU kernels are not bit-exact
across hardware, so expect recall@20 to move in the third decimal; the three seeds in `FINDINGS.md` section 6 show the
spread to compare against.

## Results (full benchmark: 8,955 product-disjoint test queries over 482,105 products)

| System | recall@10 | recall@20 | recall@50 | hit@20 |
|---|---|---|---|---|
| BM25 | 0.292 | 0.383 | 0.497 | 0.748 |
| bge-base-en-v1.5, off the shelf | 0.320 | 0.421 | 0.544 | 0.794 |
| hybrid: BM25 + bge-base (RRF) | 0.331 | 0.437 | 0.570 | 0.803 |
| bge-large-en-v1.5, off the shelf | 0.332 | 0.436 | 0.563 | 0.801 |
| Qwen3-Embedding-0.6B | 0.280 | 0.376 | 0.502 | 0.761 |
| Qwen3-Embedding-8B | 0.341 | 0.455 | 0.595 | 0.817 |
| **bge-base fine-tuned, labelled negatives (notebook 02)** | **0.356** | **0.461** | 0.585 | 0.810 |
| hybrid: BM25 + fine-tuned bge-base | 0.353 | **0.464** | **0.600** | **0.825** |

One epoch (8 minutes on an A10G) on human-labelled Substitute/Complement/Irrelevant negatives takes the 110M model
past the 70× bigger Qwen3-Embedding-8B. Mined negatives hurt (0.424): the base model's near misses are mostly
unjudged right answers.

**Stage 2 (notebook 03):** reranking the hybrid's top 20 with `bge-reranker-base` fine-tuned on the same judgements
lifts the 3 products handed to the LLM from hit@3 0.621 / nDCG@3 0.408 (stage-1 order) to **0.659 / 0.451**;
off-the-shelf rerankers manage 0.651 / 0.435 (MiniLM) and 0.648 / 0.421 (bge-reranker-base).

**How sure (notebook 02 §9):** paired bootstrap over the 8,955 test queries puts the fine-tuned model ahead of
Qwen3-8B in 98.8% of 2,000 resamples on recall@20 and 100% on recall@10; three training seeds land at
0.460 ± 0.002 recall@20. Ties are reported as ties in `FINDINGS.md`.

Total Modal spend: about $8.95, of which $5.60 is the one-time Qwen3-8B catalogue pass.
Full write-up, per-query analysis and costs in `FINDINGS.md`.

## Contents

| File | What it is |
|---|---|
| `FINDINGS.md` | Every result with the metric definitions and the design decisions. Numbers come only from executed cells or `report.json` files. |
| `01-data-and-baselines.ipynb` | The dataset and its licence, the slice (US, small version, product-disjoint split), the metrics on a worked example, the benchmark build on Modal, BM25 / dense / hybrid / bigger-model baselines, error analysis. Built by `notebooks/build_01.py`. |
| `02-fine-tuning.ipynb` | The training rows and the five negative strategies, the training setup, a smoke run, five parallel Modal runs, results against the bar, training curves, hybrid with the fine-tuned model, per-query error analysis, serving. Built by `notebooks/build_02.py`. |
| `03-reranker.ipynb` | Why a cross-encoder (and where gradient-boosted rankers fit), what the reranker sees, pointwise training on the judgements, smoke run, full runs, final-list metrics (hit@3, nDCG@3), per-query fixes and breaks, the pipeline end to end. Built by `notebooks/build_03.py`. |
| `04-spark-delta-lakehouse.ipynb` | The data pipeline again in PySpark with Delta Lake on all three locales, inside one Modal CPU container: bronze/silver/gold Delta tables, the split as semi/anti joins, passage text in Spark SQL checked against the pandas build (18 of 18 statistics, 482,105 of 482,105 texts identical) and against a pandas UDF; join plans (broadcast vs sort-merge, partition pruning); MERGE, time travel, schema enforcement, RESTORE, OPTIMIZE, VACUUM; the saved rankings re-scored with window functions (identical to `esci/metrics.py`); measured cost $0.11. |
| `SPARK.md` | The reasoning behind notebook 04: what was used (PySpark DataFrame API, Delta Lake, local mode on Modal) and why, the split as joins, the passage text in SQL and its pandas-UDF check, how to read the plans, what Delta changes, how the JARs got into the image, what verified what. |
| `esci/bootstrap.py` | Paired bootstrap confidence intervals over the test queries for every system and the differences that matter; writes `outputs/bootstrap.json`. |
| `esci/rerank_train.py` | Cross-encoder reranking of saved stage-1 rankings (`--eval-only` for off-the-shelf models) and fine-tuning (`--loss bce` pointwise or `lambda` listwise on graded gains). |
| `esci/data.py` | Download (with checksums), slice, product-disjoint split, passage text, benchmark files; `load_benchmark()` with the shared smoke-slice logic. |
| `esci/metrics.py` | recall@k, hit@k, MRR@k, graded nDCG@k over ranked lists and E/S/C/I judgements; Reciprocal Rank Fusion; ranking files. |
| `esci/embed_train.py` | Dense evaluation and fine-tuning (`sentence-transformers`, `MultipleNegativesRankingLoss`); negatives `labelled` / `substitutes` / `mined` / `random` / `none`; dev-set evaluator on a 20k sub-corpus; saves top-100 rankings. |
| `esci/baselines.py` | BM25 (`bm25s`) and hybrid (RRF of BM25 with any saved dense ranking), on CPU. |
| `jobs/modal_esci.py` | Modal app `esci-product-search`: `build_data` (CPU), `evaluate_dense` / `train_and_evaluate` / `rerank_train_and_evaluate` (GPU), `bm25_and_hybrid` (CPU); entrypoints `data`, `baselines`, `compare`, `train`, `rerank`; downloads reports and rankings to `outputs/`. |
| `jobs/modal_spark.py`, `esci/spark.py` | Modal app `esci-product-search-spark`: a CPU image with Java 17, PySpark 4.2 and the Delta 4.4 JARs baked in at build time, one `spark_pipeline` function; the Spark stages (`bronze`, `silver`, `gold`, `plans`, `delta`, `score`) as plain functions of a `SparkSession`, report checkpointed after every stage. Nothing runs on the laptop. |
| `notebooks/run_notebook.py` | Executes a notebook cell by cell, writing the file and a timestamped log after every cell (`--start`, `--until` to resume). |
| `data/raw/`, `data/processed/` | Downloaded and derived data; gitignored. |
| `outputs/`, `logs/` | Reports, rankings, models brought back from Modal; execution logs; gitignored. |

## Licence

Code: MIT (`LICENSE`). Data: the Amazon Shopping Queries Dataset is Apache-2.0, downloaded at run time from
[amazon-science/esci-data](https://github.com/amazon-science/esci-data); nothing from it is redistributed here.

## Modal resources

Volumes `esci-data` (raw files + benchmark), `esci-runs` (reports, rankings, fine-tuned models) and `hf-cache`
(Hugging Face model weights: `bge-base`, `bge-large`, `Qwen3-Embedding-0.6B/8B` download once and are reused by every run).
All three are created on first use.
The Delta lake of notebook 04 lives in `esci-data/lake/<run-name>/` (about 1.6 GB per full run; `spark-smoke` and `spark-all`
can be deleted with `modal volume rm esci-data lake/<run-name> -r` once the reports are in `outputs/`).
