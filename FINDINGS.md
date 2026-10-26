# Findings

Every number here comes from an executed cell of a notebook in this folder or from a `report.json` in
`outputs/`; nothing is estimated unless it says so. Headline: a fine-tuned 110M embedding model reaches 0.461 recall@20
on 8,955 unseen-product queries over 482k products, above Qwen3-Embedding-8B off the shelf (0.455), 70× its size;
hybrid with BM25 takes stage 1 to 0.464, and a reranker fine-tuned on the same judgements lifts the top-3 from
hit@3 0.621 to 0.659 (nDCG@3 0.408 → 0.451). Nothing was trained or encoded on the laptop.

## 0. Design decisions, fixed before any result

- **Pipeline shape.** Embedding model (stage 1) → cross-encoder reranker over **k = 20** candidates → **3**
  products into the LLM. Stage 1 is judged on **recall@20**; recall@10 and recall@50 are reported alongside.
  MRR and nDCG belong to the reranker and the final list and are only diagnostics here.
- **Slice.** Locale US (English-only models and stemmer), the "small" (competition, harder) version:
  29,844 queries, 601,354 judgements, 482,105 products.
- **Split.** The official query split, made **product-disjoint**: every product that appears in any test query's
  judged list is removed from the training judgements. 500 training queries are held out as a dev set.
- **Corpus.** All 482,105 products of the slice, training products included. The passage is title + brand/colour
  line + bullet points, cut at 1,200 characters; the description is left out in this version.
- **Labels.** Exact = relevant for recall. Substitute / Complement / Irrelevant are the labelled hard negatives
  for training (to be compared with mined and random negatives). Graded nDCG uses the KDD Cup gains
  E 1.0 / S 0.1 / C 0.01 / I 0.
- **Unjudged products** returned by a system count as not relevant (pooled evaluation). Fair between systems,
  a floor in absolute terms.

## 1. The dataset (notebook 01, §2-3)

| | queries | judgements | distinct products |
|---|---|---|---|
| All locales, large version | 130,652 | 2,621,288 | 1,802,772 |
| US, large | 97,345 | 1,818,825 | |
| US, small (used here) | 29,844 | 601,354 | 482,105 |

- Label shares over all 2.6M judgements: E 65.2%, S 21.9%, I 10.0%, C 2.9%. In the US small version:
  E 43.5%, S 35.1%, I 16.9%, C 4.5%. The small version is harder because the easy queries were removed.
- US small: 20.1 judged products per query on average (median 16, max 188); 8.8 Exact per query (median 7).
  One test query has no Exact product and is dropped.
- Queries are short: 3.5 words on average, 29 at most. Brands, model numbers and misspellings are common.

### The product-disjoint split

| | queries | judgements | distinct products | Exact pairs | Exact per query |
|---|---|---|---|---|---|
| train (after removal, minus dev) | 20,039 | 341,147 | 308,055 | 153,467 | 7.66 |
| dev | 500 | 8,621 | 8,584 | 3,762 | 7.52 |
| test (frozen) | 8,955 | 181,686 | 164,887 | 79,708 | 8.90 |

- 34,756 of the 164,900 products in test lists (21.1%) also appear in training lists; among test Exact
  products the share is 20.7%. Removing those products from training drops 66,945 of 419,653 training
  judgements (16.0%) and 338 queries that lose all their Exact products. After the removal, zero test products
  appear in the training judgements.

## 2. Baselines

### 2a. Smoke slice (notebook 01, §6): 500 random test queries, corpus of 20,000 products

The corpus is the 500 queries' judged products (about 10k) plus random fillers. It is a 24× smaller haystack
than the full benchmark, so these numbers are higher than the full ones will be; they exist to check the pipeline
and to price the full runs. Rankings saved to depth 100 for every system.

| System | Params | recall@10 | recall@20 | recall@50 | hit@20 | MRR@20 | nDCG@20 | Corpus encode |
|---|---|---|---|---|---|---|---|---|
| BM25 (bm25s, stopwords + Snowball stemmer) | 0 | 0.500 | 0.663 | 0.784 | 0.920 | 0.678 | 0.586 | 1.6 s (CPU) |
| bge-base-en-v1.5, off the shelf | 110M | 0.543 | 0.717 | 0.851 | 0.950 | 0.725 | 0.637 | 14.9 s (A10G) |
| hybrid: BM25 + bge-base, Reciprocal Rank Fusion | – | 0.546 | 0.712 | 0.861 | 0.946 | 0.713 | 0.632 | – |
| Qwen3-Embedding-0.6B, shopping instruction | 0.6B | 0.515 | 0.688 | 0.821 | 0.940 | 0.681 | 0.603 | 54 s (L40S) |
| Qwen3-Embedding-8B, shopping instruction | 7.6B | 0.577 | 0.761 | 0.885 | 0.944 | 0.739 | 0.676 | 377 s (L40S) |

Early reads (500 queries, so differences under about 0.02 are noise):

- BM25 is a serious baseline on shopping queries: 0.663 recall@20 against 0.717 for the dense model.
- Qwen3-Embedding-0.6B is *worse* than bge-base here; the 8B model is better by 0.044
  recall@20 at 70× the parameters and 25× the encode time.
- Hybrid fusion did not beat the dense model alone on recall@20 on this slice (0.712 vs 0.717); it helped
  recall@50. Whether that holds on the full corpus, where BM25's exact-word matches matter more against a
  bigger haystack, is one of the questions for the full run.
- Passages: median 146 bge tokens, mean 152, 19.6% longer than the 256-token limit (from a 10k sample).

### Measured Modal costs so far (container time × list price; the dashboard invoice is the authority)

| Job | Container | GPU / CPU | Estimated cost |
|---|---|---|---|
| Data build: download 1.1 GB (26 s), slice, write benchmark | 59 s | 4 CPU cores, 16 GB | $0.005 |
| Smoke: bge-base (fp16) | 29 s | A10G | $0.009 |
| Smoke: Qwen3-Embedding 0.6B + 8B (bf16) | 465 s | L40S | $0.25 |
| Smoke: BM25 + hybrid | 5 s | 8 CPU cores | $0.001 |
| A first smoke attempt that failed on download (bge-base in fp32) | 106 s | A10G | $0.03 |

Loading evaluation models in half precision made bge-base 4.7× faster per product (3.5 → 0.7 s per 1k).

### Priced from the smoke throughput, full corpus of 482,105 products and 8,955 test queries

| Model | s per 1k products | Full corpus | Estimated cost |
|---|---|---|---|
| bge-base-en-v1.5 on A10G | 0.7 | ~6 min | $0.11 |
| bge-large-en-v1.5 on A10G (about 3× bge-base) | ~2.2 | ~18 min | ~$0.35 |
| Qwen3-Embedding-0.6B on L40S | 2.7 | ~22 min | $0.71 |
| Qwen3-Embedding-8B on L40S | 18.9 | ~152 min | $4.94 |

### 2b. Full benchmark (notebook 01, §6-7): 8,955 product-disjoint test queries, corpus of 482,105 products

From `outputs/baselines-us-small/`, `outputs/baselines-us-small-bm25/` and `outputs/compare-qwen/` (`report.json`
and top-100 rankings per query in each).

| System | Params | recall@10 | **recall@20** | recall@50 | hit@20 | MRR@20 | nDCG@20 | Corpus encode |
|---|---|---|---|---|---|---|---|---|
| BM25 (lexical, no model) | 0 | 0.292 | 0.383 | 0.497 | 0.748 | 0.465 | 0.337 | 66 s (8 CPU cores) |
| bge-base-en-v1.5, off the shelf | 110M | 0.320 | 0.421 | 0.544 | 0.794 | 0.510 | 0.370 | 355 s (A10G) |
| hybrid: BM25 + bge-base, Reciprocal Rank Fusion | – | 0.331 | 0.437 | 0.570 | 0.803 | 0.525 | 0.385 | – |
| bge-large-en-v1.5, off the shelf | 335M | 0.332 | 0.436 | 0.563 | 0.801 | 0.520 | 0.384 | 1,136 s (A10G) |
| Qwen3-Embedding-0.6B, shopping instruction | 0.6B | 0.280 | 0.376 | 0.502 | 0.761 | 0.467 | 0.332 | 1,288 s (L40S) |
| **Qwen3-Embedding-8B, shopping instruction** | 7.6B | **0.341** | **0.455** | **0.595** | **0.817** | **0.532** | **0.401** | 8,979 s (L40S) |

This is the bar for the fine-tuned 110M model in notebook 02: **0.421 is where it starts, 0.455 is the best
off-the-shelf number, and 0.437 is what hybrid fusion gives for free.**

Findings:

1. **The haystack matters.** From the 20k-product smoke slice to the 482k corpus, bge-base drops from 0.717
   to 0.421 recall@20 and every system loses 0.28 to 0.31. Smoke slices price runs; they do not rank systems.
2. **Hybrid is worth it on the full corpus.** RRF of BM25 and bge-base (0.437) beats both parts on every cutoff
   and matches bge-large (0.436) at a third of the encode time. Per query (recall@20): dense beats BM25 on 34.2%,
   BM25 beats dense on 24.0%, tie on 41.8%. BM25 wins outright on model numbers and odd tokens ("aeramax 290",
   "aga stove", "11.5 x 8 clear plastic bux"); the dense model wins on paraphrase and intent ("8 month old
   towels", "49ers gym apparel", "4\" non marking caster wheel").
3. **Fusion is a blunt tool.** Hybrid beats both parts on only 11.5% of queries and is worse than the better part
   on 28.5%: Reciprocal Rank Fusion averages ranks, so when one retriever is right and the other is lost, the
   right one is diluted. That is the argument for a reranker over the union rather than a fixed fusion rule.
4. **Off-the-shelf size buys little.** bge-large: 3× the parameters, 3.2× the encode time, +0.015 recall@20.
   Qwen3-Embedding-0.6B is below BM25 (0.376). Qwen3-Embedding-8B is the best off-the-shelf model at
   +0.034 over bge-base, for 25× the encode time (2.5 hours on an L40S for one pass over the catalogue) and a
   4,096-float vector instead of 768.
5. **14.1% of test queries get no Exact product in anyone's top 20** (BM25, dense or hybrid). Those are the
   queries to inspect for annotation gaps and for where fine-tuning cannot help.
6. **Absolute numbers are a floor** by construction: only about 20 products per query were ever judged, so a
   good unjudged product in the top 20 counts as a miss for every system equally.

### Measured Modal cost of notebook 01 (container seconds × list price; the Modal dashboard is the authority)

| Job | Container | Hardware | Estimated cost |
|---|---|---|---|
| Data build (download 1.1 GB, slice, write benchmark) | 59 s | 4 CPU cores, 16 GB | $0.005 |
| Smoke: bge-base; Qwen3 0.6B + 8B; BM25 + hybrid; one failed first attempt | 29 s; 465 s; 5 s; 106 s | A10G; L40S; CPU; A10G | $0.29 |
| Full: bge-base + bge-large | 1,516 s | A10G | $0.46 |
| Full: BM25 + hybrid | 95 s | 8 CPU cores | $0.02 |
| Full: Qwen3-Embedding 0.6B + 8B | 10,329 s | L40S | $5.60 |
| **Total** | | | **about $6.40** |

The 8B run was priced at $4.94 + $0.71 from the smoke throughput and came in at $5.60: the per-product
encode speed held (18.6 vs 18.9 s per 1k products).

## 3. The metrics, in plain words

Computed per query over the whole corpus, then averaged. A query has several Exact products (9 on average).

- **recall@k**: share of the query's Exact products in the top k. The stage-1 headline, with k = 20.
- **hit@k** (accuracy@k): 1 if at least one Exact product is in the top k.
- **MRR@k**, Mean Reciprocal Rank: 1 / rank of the first Exact product, 0 if none in the top k.
- **nDCG@k**, normalized Discounted Cumulative Gain: gain / log2(rank + 1) summed over the top k, divided by
  the ideal value; graded gains E 1.0, S 0.1, C 0.01, I 0. The reranker's metric.

Worked example (notebook 01, §4): query "yellow pepper", 4 Exact products; a toy ranking with Exact products
at ranks 2, 5 and 10, a Substitute at rank 1 and unjudged products elsewhere scores recall@5 0.5,
recall@10 0.75, MRR 0.5, hit 1, nDCG@10 0.527.

## 4. Fine-tuning (notebook 02): the 110M model beats the 8B model

Setup: `bge-base-en-v1.5`, full fine-tune, `MultipleNegativesRankingLoss`, 1 epoch over 54,880 rows (20,039
product-disjoint training queries × up to 3 Exact products), batch 64, learning rate 2e-5, 256 tokens, one A10G
per run. Five runs differ only in the negative paired with each row. All scored on the frozen test set over the
full 482,105-product corpus with the notebook 01 evaluator. Reports: `outputs/embed-v1-*/report.json`.

| System | Params | recall@10 | **recall@20** | recall@50 | hit@20 | MRR@20 | nDCG@20 |
|---|---|---|---|---|---|---|---|
| bge-base, off the shelf | 110M | 0.320 | 0.421 | 0.544 | 0.794 | 0.510 | 0.370 |
| hybrid: BM25 + bge-base | – | 0.331 | 0.437 | 0.570 | 0.803 | 0.525 | 0.385 |
| bge-large, off the shelf | 335M | 0.332 | 0.436 | 0.563 | 0.801 | 0.520 | 0.384 |
| Qwen3-Embedding-8B, off the shelf | 7.6B | 0.341 | 0.455 | 0.595 | 0.817 | 0.532 | 0.401 |
| bge-base fine-tuned, pairs only (`none`) | 110M | 0.333 | 0.439 | 0.567 | 0.800 | 0.525 | 0.390 |
| bge-base fine-tuned, random negatives | 110M | 0.336 | 0.440 | 0.571 | 0.802 | 0.526 | 0.391 |
| bge-base fine-tuned, mined negatives | 110M | 0.323 | 0.424 | 0.542 | 0.776 | 0.501 | 0.372 |
| bge-base fine-tuned, Substitute negatives | 110M | **0.356** | **0.461** | 0.587 | 0.815 | **0.549** | 0.410 |
| **bge-base fine-tuned, labelled S/C/I negatives** | 110M | **0.356** | **0.461** | 0.585 | 0.810 | **0.549** | **0.411** |
| hybrid: BM25 + bge-base fine-tuned (Substitute run) | – | 0.353 | **0.464** | **0.600** | **0.825** | 0.550 | 0.410 |

Dev-set recall@20 (500 queries, 20k sub-corpus) ranked the runs the same way as the test set: labelled 0.784,
substitutes 0.785, random 0.775, none 0.775, mined 0.758.

Findings:

1. **A fine-tuned 110M model beats the best off-the-shelf model, which is 70× bigger**: 0.461 vs 0.455 recall@20,
   0.356 vs 0.341 recall@10, 0.549 vs 0.532 MRR@20, after one epoch (8 minutes of training) on human-labelled
   negatives. Per query it is better than Qwen3-8B on 30.8%, worse on 28.0%, equal on 41.2%: the two are close
   in aggregate but disagree on most of the hard queries. Qwen3-8B keeps a small edge at recall@50 (0.595 vs 0.587)
   and on queries with 11 or more Exact products; the fine-tuned model wins on queries with 1 to 10.
2. **Labelled hard negatives are the whole gain over the controls.** Pairs only 0.439, random 0.440, labelled
   0.461: the extra +0.021 comes from *which* negative, not from having one. Substitutes only (0.461, with 15.6%
   of rows falling back to random when a query has no Substitute) ties with the full S/C/I mix: the Substitutes are
   doing the work; Complements and Irrelevants add little.
3. **Mined negatives hurt** (0.424, barely above the untrained 0.421, and below pairs only at 0.439). The base
   model's top-30 for a shopping query is full of unjudged products that are in fact Exact matches (only about 20 of
   482k products were judged per query), so mining teaches the model to push right answers away. On a dataset with
   dense human labels, use them; on one without, mined negatives need a filter (a cross-encoder or margin rule)
   before they are safe.
4. **Fine-tuning helps where the base model was weakest.** Per query vs base: better on 32.8%, worse on 16.6%;
   the share of test queries with no Exact product in the top 20 falls from 20.5% to 18.5% (Qwen3-8B 18.3%,
   BM25 25.2%). The losses are single-product queries that flip (model numbers like "bt-705", "batman dvd"),
   the kind BM25 or a reranker over the union recovers.
5. **Hybrid still adds a little on top of the fine-tuned model** (+0.003 recall@20, +0.013 recall@50, +0.010
   hit@20): less than the +0.016 it added to the base model, because the dense side now covers more of what BM25
   used to contribute. It is now the best stage-1 system at 0.464 / 0.600 / 0.825.
6. **The training loss is not comparable across negative kinds**: random and none converge to 0.3-0.45, the hard
   kinds to 0.9-1.0, because hard negatives make the batch classification harder by design. Only the dev and test
   metrics rank the runs.
7. **The evaluation is reproducible**: the `before` row re-computed inside the labelled run matched notebook 01's
   bge-base numbers exactly (0.421 / 0.544 / 0.794).

### Measured Modal cost of notebook 02

| Run | Training | Container | Estimated cost |
|---|---|---|---|
| smoke (20 steps, smoke slice) | 29 s | 88 s | $0.03 |
| embed-v1-labelled (with before-eval) | 489 s | 1,245 s | $0.38 |
| embed-v1-mined (with before-eval + mining) | 502 s | 1,301 s | $0.40 |
| embed-v1-substitutes | 485 s | 881 s | $0.27 |
| embed-v1-random | 484 s | 875 s | $0.27 |
| embed-v1-none (two columns, faster) | 290 s | 673 s | $0.21 |
| hybrid with the best run (CPU) | – | 76 s | $0.01 |
| **Total** | | | **about $1.57** (estimate was $1.75) |

Project total so far: about $8.00 at list price, of which $5.60 is the one-time Qwen3-8B catalogue pass.

## 5. Stage 2: the cross-encoder reranker (notebook 03)

Setup: the top 20 candidates per test query from the best stage-1 system (hybrid BM25 + fine-tuned bge-base,
recall@20 0.464), re-ordered by a cross-encoder. `BAAI/bge-reranker-base` (278M) fine-tuned for one epoch on
144,806 pointwise rows from the training judgements (Exact = 1; Substitute / Complement / Irrelevant = 0; up to 3
positives and 5 negatives per query; binary cross-entropy with class weighting; batch 64; 320 tokens). Compared
with the stage-1 order itself and two off-the-shelf rerankers. Metrics on the re-ordered 20; recall@20 is fixed
at 0.464 by stage 1. Reports: `outputs/rerank-v1/`, `outputs/rerank-offtheshelf/`.

| Final-list system | Params | recall@3 | **hit@3** | MRR@3 | **nDCG@3** | nDCG@10 |
|---|---|---|---|---|---|---|
| stage-1 order (hybrid BM25 + fine-tuned bge-base), no reranker | – | 0.181 | 0.621 | 0.520 | 0.408 | 0.383 |
| MiniLM-L-6 MS MARCO cross-encoder, off the shelf | 23M | 0.193 | 0.651 | 0.550 | 0.435 | 0.403 |
| bge-reranker-base, off the shelf | 278M | 0.187 | 0.648 | 0.537 | 0.421 | 0.396 |
| **bge-reranker-base, fine-tuned on the ESCI judgements** | 278M | **0.203** | **0.659** | **0.567** | **0.451** | **0.414** |

Findings:

1. **The fine-tuned reranker improves the 3 slots on every metric**: hit@3 0.621 → 0.659 (+0.038), MRR@3
   0.520 → 0.567, nDCG@3 0.408 → 0.451. Per query it fixes hit@3 on 10.8% and breaks it on 6.9%; on nDCG@3 it is
   better on 35.3%, worse on 25.9%.
2. **Off-the-shelf rerankers help less than expected on shopping queries.** MiniLM (23M, trained on MS MARCO
   web questions) gains +0.027 nDCG@3; bge-reranker-base off the shelf only +0.013 and is below the 12× smaller
   MiniLM. Relevance in product search is attribute matching ("without rims", "not digital"), not question
   answering; the fine-tune on labelled judgements is what teaches that.
3. **The ceiling is stage 1.** 82.5% of test queries have at least one Exact product among their 20 candidates,
   so hit@3 cannot exceed 0.825 here; the fine-tuned reranker reaches 0.659. The remaining headroom is split
   between a better reranker and better recall.
4. **Fixes and breaks are both about fine attribute constraints.** Fixed: "canon t8i battery", "john deere d140
   drive belt", "hidden shelves". Broken: "compact projector", "jansport big student backpack", "dog hip joint
   c.f.". Pointwise binary training treats a Substitute like an Irrelevant; the graded listwise loss (`--loss
   lambda`) is the next experiment for exactly these.
5. **The dev-set reranking metric** (reranking each dev query's own judged list) reached nDCG@3 0.737 and MRR@3
   0.833 after the epoch: far above the test numbers because the judged list is only ~20 products with ~8 Exact,
   a much easier task than reranking 20 retrieved candidates where many Exact products are missing.

### The pipeline end to end

| Stage | System | Judged on | Off the shelf | Fine-tuned on ESCI |
|---|---|---|---|---|
| 1. Retrieval over 482,105 products | hybrid BM25 + bge-base (110M) | recall@20 | 0.437 (bge-base alone 0.421; Qwen3-8B 0.455) | **0.464** |
| 2. Rerank the 20 | bge-reranker-base (278M) | hit@3 / nDCG@3 | 0.648 / 0.421 | **0.659 / 0.451** |
| 3. Answer | LLM over the 3 products | not measured here | | |

### Measured Modal cost of notebook 03

| Run | Training | Container | Estimated cost |
|---|---|---|---|
| rerank-smoke (30 steps, 500 queries) | 18 s | 71 s | $0.02 |
| rerank-v1 (bge-reranker-base: before, 1 epoch on 144,806 rows, after) | 862 s | 1,214 s | $0.37 |
| rerank-offtheshelf (MiniLM + stage-1 order on 8,955 × 20 pairs) | – | 63 s | $0.02 |
| **Total** | | | **about $0.41** (estimate was $0.75) |

Project total: about $8.40 at list price for three notebooks, of which $5.60 is the one-time Qwen3-8B pass.

## 6. How sure are we: bootstrap intervals and extra seeds (notebook 02, §9)

The official test split is never touched, so every number stays comparable with published ESCI results.
Two checks measure the two sources of noise separately. `esci/bootstrap.py` → `outputs/bootstrap.json`.

### Paired bootstrap over the 8,955 test queries (2,000 resamples, 95% intervals)

The test queries are redrawn with replacement, the saved per-query scores are averaged on each redraw (no
model is re-run), and every system is scored on the same redraw ("paired").

| System | recall@20 | 95% interval |
|---|---|---|
| BM25 | 0.383 | 0.376 – 0.390 |
| bge-base, off the shelf | 0.421 | 0.413 – 0.428 |
| hybrid: BM25 + bge-base | 0.437 | 0.430 – 0.444 |
| bge-large, off the shelf | 0.436 | 0.429 – 0.443 |
| Qwen3-Embedding-0.6B | 0.376 | 0.369 – 0.383 |
| Qwen3-Embedding-8B | 0.455 | 0.448 – 0.462 |
| fine-tuned, pairs only | 0.439 | 0.432 – 0.446 |
| fine-tuned, random negatives | 0.440 | 0.432 – 0.447 |
| fine-tuned, mined negatives | 0.424 | 0.416 – 0.431 |
| fine-tuned, Substitute negatives | 0.461 | 0.454 – 0.469 |
| fine-tuned, labelled negatives | 0.461 | 0.454 – 0.468 |
| hybrid: BM25 + fine-tuned | 0.464 | 0.456 – 0.471 |

Differences that matter (a − b, observed, 95% interval, share of resamples with a ahead):

| a | b | diff | interval | a ahead |
|---|---|---|---|---|
| fine-tuned, labelled | Qwen3-Embedding-8B | +0.006 | +0.001 – +0.011 | 98.8% |
| hybrid: BM25 + fine-tuned | Qwen3-Embedding-8B | +0.008 | +0.003 – +0.014 | 99.8% |
| fine-tuned, labelled | Qwen3-Embedding-8B, **recall@10** | +0.015 | +0.011 – +0.020 | 100% |
| fine-tuned, labelled | bge-base off the shelf | +0.040 | +0.036 – +0.044 | 100% |
| fine-tuned, labelled | hybrid BM25 + bge-base | +0.024 | +0.019 – +0.028 | 100% |
| fine-tuned, labelled | fine-tuned, random negatives | +0.021 | +0.018 – +0.024 | 100% |
| fine-tuned, labelled | fine-tuned, mined negatives | +0.037 | +0.034 – +0.041 | 100% |
| fine-tuned, labelled | fine-tuned, Substitute negatives | −0.000 | −0.003 – +0.002 | 38% (tie) |
| fine-tuned, mined | bge-base off the shelf | +0.003 | −0.002 – +0.008 | 88% (tie) |
| hybrid: BM25 + fine-tuned | fine-tuned, Substitute | +0.002 | −0.002 – +0.006 | 85% (tie) |
| hybrid: BM25 + bge-base | bge-large off the shelf | +0.001 | −0.003 – +0.005 | 70% (tie) |
| BM25 | Qwen3-Embedding-0.6B | +0.007 | +0.001 – +0.013 | 98.4% |

Stage 2 (hit@3 / nDCG@3): the fine-tuned reranker is ahead of the stage-1 order in 100% of resamples on both
(hit@3 +0.039, interval +0.030 – +0.047; nDCG@3 +0.043, +0.037 – +0.049), ahead of MiniLM off the shelf in 98.1%
on hit@3 (+0.008, +0.000 – +0.016) and 100% on nDCG@3 (+0.016, +0.011 – +0.021).

What this settles:

- **"Fine-tuned 110M beats Qwen3-8B" holds on the test side, narrowly on recall@20** (ahead in 98.8% of
  resamples, interval just above zero) **and clearly on recall@10** (100%, +0.015). The hybrid with the fine-tuned
  model is ahead on recall@20 in 99.8%. The safe sentence for a write-up is "matches or beats, and clearly beats
  at the top of the list".
- **Everything else in the negatives story is far outside noise**: labelled vs random +0.021, vs pairs only
  +0.022, vs mined +0.037, all 100%.
- **Three claims are ties and should be written as such**: labelled vs Substitutes only; mined negatives vs
  no fine-tuning; hybrid vs bge-large off the shelf. The small hybrid gain on top of the fine-tuned model
  (+0.002) is also within noise.

### Extra seeds of the labelled configuration

Same data and settings, seeds 42 (the original), 43 and 44, scored on the same frozen test set. The seed decides
which 3 Exact products each query keeps, which negative each row gets, the batch order and dropout.
`outputs/embed-v1-labelled-seed43/`, `outputs/embed-v1-labelled-seed44/`.

| Seed | recall@10 | recall@20 | recall@50 | hit@20 | nDCG@20 |
|---|---|---|---|---|---|
| 42 | 0.3561 | 0.4610 | 0.5851 | 0.8101 | 0.4109 |
| 43 | 0.3540 | 0.4597 | 0.5845 | 0.8114 | 0.4089 |
| 44 | 0.3539 | 0.4580 | 0.5843 | 0.8119 | 0.4083 |
| **mean ± std** | 0.3547 ± 0.0012 | **0.4596 ± 0.0015** | 0.5846 ± 0.0004 | | |

- Training-side noise is small: a spread of 0.003 in recall@20 across seeds, smaller than the test-side interval
  (about ±0.007 on a single system, ±0.005 on a paired difference).
- All three seeds are above Qwen3-Embedding-8B (0.4551) on recall@20; the lowest margin is +0.003, which on its own
  is inside the paired interval. On recall@10 every seed is at least +0.013 ahead, well outside it.
- The seed-42 run that the tables quote is the best of the three by 0.001 to 0.003: quote the mean, 0.460.

**The sentence that survives both checks:** a 110M model fine-tuned for 8 minutes on human-labelled hard
negatives scores 0.460 ± 0.002 recall@20 across seeds against 0.455 for Qwen3-Embedding-8B off the shelf,
matching or narrowly beating it on recall@20 and clearly beating it on recall@10 (0.355 vs 0.341), at 1/70 of the
parameters and 1/25 of the encode time.

### Cost of the validation step

Bootstrap: $0 (local). Two seed runs: 884 s + 895 s on an A10G, about $0.54. Project total about $8.95 at list
price (plus Modal's billing overheads not counted here).

## 7. Open

- A LambdaMART (LightGBM, CPU) over the per-pair scores (BM25, dense, cross-encoder, ranks): the production
  shape of the final ranker, and the answer in numbers to "isn't the reranker just a GBDT".
- Listwise reranker training (`--loss lambda`) on the graded labels; reranker training rows drawn from the
  fine-tuned retriever's own candidates.
- Stage 1: a second epoch and a bigger batch for the labelled run; filtered mined negatives mixed with labelled.
- Spanish and Japanese with multilingual models; the large version; passages with the description.

## 8. The data pipeline in Spark with Delta Lake, all three locales (notebook 04)

**What and why.** Notebook 01 built the US benchmark with pandas in 23 seconds; 1.1 GB of Parquet is not big data.
Notebook 04 rebuilds the data layer with PySpark 4.2.0 and Delta Lake 4.4.1 for `us`, `es` and `jp` inside one Modal CPU
container (8 cores / 32 GB, Spark local mode pinned to `local[8]`; the container reports 24 host CPUs but is billed
and throttled for 8), to learn the API, the plans and the Delta transaction log on data whose right answers are
known. Nothing runs on the laptop. Reports: `outputs/spark-smoke/report.json` (US only), `outputs/spark-all/report.json`.

**Verification against the pandas build (US, small version)**

| Check | Result |
|---|---|
| Split statistics (`stats.json` of notebook 01) | 18 of 18 identical (one count reconciled: notebook 01's 338 training queries dropped after overlap removal excludes 11 queries whose every judgement was removed; 349 is the full number and 20,888 − 349 = 20,539 kept in both builds) |
| Passage text, Spark SQL vs `corpus.parquet` | 482,105 of 482,105 products byte-identical |
| Passage text, Spark SQL vs `esci.data.product_text` in a pandas UDF | identical for all products in all locales (us 482,105; es 167,761; jp 233,850) |
| Saved rankings re-scored with window functions vs `esci/metrics.py` | max absolute difference 0.0000 on recall@10/20/50, hit@20, nDCG@20 for bge-base off the shelf, the fine-tuned model and the hybrid |

The dev subset is drawn with Spark's `rand(42)` and differs in *membership* from numpy's `default_rng(42)` draw;
train+dev totals match exactly.

**The three locales (small version, after the product-disjoint split)**

| Locale | Queries | Judgements | Products | Train q | Dev q | Test q | Test Exact pairs | Train judgements removed (overlap) |
|---|---|---|---|---|---|---|---|---|
| us | 29,844 | 601,354 | 482,105 | 20,039 | 500 | 8,955 | 79,708 | 66,945 (16.0%) |
| es | 8,049 | 218,774 | 167,761 | 4,999 | 500 | 2,417 | 27,074 | 29,362 (19.2%) |
| jp | 10,407 | 297,883 | 233,850 | 6,679 | 500 | 3,123 | 41,597 | 36,556 (17.5%) |

No test product appears in any locale's final training set. Passage text: mean 653 / 665 / 287 characters
(us / es / jp), share cut at 1,200 characters 0.207 / 0.242 / 0.002; brand present 0.944 / 0.950 / 0.828.
Judged products per query: mean 18.0 / 23.8 / 25.3, Exact per query 8.0 / 10.2 / 12.0; the most-judged query is
"airpods" (us, 134 judged, 42 Exact). Label share over all judgements (all versions): Exact 0.686 (us), 0.567 (es),
0.580 (jp); the small version is the harder subset, hence the lower Exact shares in the benchmark (§1).

**Execution model (`plans` stage, all locales).** The judgements ⋈ products join (980,064 rows, 3.2 s) was *not* a
sort-merge join: after column pruning the 14.7 MB `silver_judgements` table fell under the 10 MB
`autoBroadcastJoinThreshold` estimate, so Spark broadcast the judgements and streamed the 1.1 GB products table
once with no shuffle of it; the only `Exchange` on that side is the `dropDuplicates`. The product-disjoint anti
join (643,706 rows) is a `BroadcastHashJoin LeftAnti` (1.3 s); with broadcasting disabled it becomes a
`SortMergeJoin` with both sides shuffled (1.5 s). A filter on `product_locale` shows up as a `PartitionFilters`
entry in the scan (partition pruning).

**Delta Lake (`delta` stage, on a copy of the US judgements).** Version 0: 531,454 rows, 79,708 Exact test pairs.
One `MERGE` (2.9 s) applied a simulated re-judging batch, 1,000 Substitute test pairs relabelled Exact and 500 new
judgements inserted: 531,954 rows, 80,708 Exact. `versionAsOf 0` still reads 79,708 (time travel). An append with
`esci_label` as an integer was refused (`DELTA_FAILED_TO_MERGE_FIELDS`, schema enforcement). `RESTORE` to version 0
brought back 79,708 / 531,454 as version 2. `OPTIMIZE` compacted 8 files into 1, `VACUUM` (0 h retention, a demo
setting) removed the rest: 14 → 15 → 1 Parquet files, 11.3 → 17.9 → 6.5 MB.

**Measured Modal cost** (8 cores + 32 GB = $0.632 per hour at list price)

| Run | Locales | bronze | silver | gold | plans | delta | score | Container s | Cost |
|---|---|---|---|---|---|---|---|---|---|
| spark-smoke | us | 71 s | 178 s | 5 s | 6 s | 28 s | 13 s | 262 | $0.046 |
| spark-all | us, es, jp | 96 s | 200 s | 6 s | 7 s | 27 s | 11 s | 376 | $0.066 |

Plus two failed smoke attempts (about $0.04) and three short image builds. About $0.15 in total for the notebook;
running project total about $9.10 at list price.

**Gotchas met on the way**

- The Delta JARs are not in the `delta-spark` pip package. Baking them into the image means copying only the
  `io.delta_*` and `io.unitycatalog_*` JARs Ivy resolves: it also resolves Jackson 2.13, Hadoop, Parquet and log4j
  JARs that Spark 4.2 already ships at newer versions, and two versions of one library in `jars/` load in
  directory order, which differs between containers (the first smoke run passed, the second failed with
  `NoSuchMethodError: JsonFactory.setStreamReadConstraints`).
- `pandas_udf` infers the UDF kind from the type hints; with `from __future__ import annotations` they are strings
  and it raises `UNSUPPORTED_SIGNATURE`. Attach real `pd.Series` objects via `__annotations__`.
- Python's `str.split()` treats the ideographic space U+3000 as whitespace; Java's `\s` does not unless the regex
  starts with `(?U)`. With it, the Japanese passage text matched the Python function on every product.
- `local[*]` on Modal would start 24 task slots for an 8-core quota; pin `local[<requested cores>]`.
- Delta on a Modal Volume works as a plain file system; commit the Volume after each stage.

**Open.** Spark Connect from the laptop (thin `pyspark-client`, no local JVM) against a server in a Modal container;
Databricks Free Edition for the same tables under Unity Catalog.
