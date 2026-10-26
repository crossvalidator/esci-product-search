# The Spark work, explained

This file explains the reasoning behind notebook 04 (`04-spark-delta-lakehouse.ipynb`) and the two scripts it
runs, `esci/spark.py` and `jobs/modal_spark.py`. The notebook shows the results; this is the "why" behind the design,
written for someone who has used pandas and wants to understand what Spark changes.

## 1. What was used

**PySpark**, the Python API of Apache Spark, version 4.2.0. Nothing in Scala or Java was written. Spark itself is
a Java Virtual Machine (JVM) program; `pyspark` is a Python package that starts that JVM, keeps a connection to it,
and translates every DataFrame call into a plan the JVM executes. The Python process never touches the rows
unless you explicitly ask it to (the one place that happens here is the pandas UDF in section 5).

Within PySpark, three layers were available and two were used:

| Layer | Used? | Why |
|---|---|---|
| **DataFrame API** (`df.filter`, `df.join`, `df.groupBy`, `F.regexp_replace`, window functions) | yes, everywhere | It is the modern API. Spark's optimiser (Catalyst) sees the whole query and chooses joins, pushes filters down and prunes columns. This is what you would write on a job. |
| **Spark SQL strings** (`spark.sql("SELECT …")`) | no | Same engine as the DataFrame API, same plans. The DataFrame form composes better in Python functions and keeps the column names checked by the IDE. Either is fine; mixing them in one project is noise. |
| **RDDs** (Resilient Distributed Datasets, `sc.parallelize`, `rdd.map`) | no | The original low-level API: Python functions applied to Python objects, row by row, with no optimiser. Slow from Python and unnecessary for tabular data. |

**Delta Lake** 4.4.1 through the `delta-spark` Python package. The package is a thin wrapper: the actual
implementation is a set of JARs that Spark loads. Section 7 covers what Delta adds and section 8 how the JARs
got into the image.

**Where it runs.** Spark in *local mode* inside one Modal container: `master("local[8]")` means one JVM that acts as
driver and executor at once, with 8 task slots. There is no cluster, no cluster manager and no network shuffle
between machines. Everything else is the same as on a cluster: jobs, stages, tasks, shuffles, plans and the Delta
log all behave identically, which is why local mode is the right place to learn them. The laptop never starts a
JVM; it calls one Modal function and reads a JSON report back.

## 2. Why Spark at all, on 1.1 GB

It is not needed. The whole ESCI dataset is 2.6 million judgement rows (51 MB) and 1.8 million product rows
(1.1 GB), and notebook 01 built the US benchmark with pandas in 23 seconds. The reason to do it in Spark is the
same reason people learn Spark on a few million NYC taxi trips: the code and the execution model are
identical at 10 TB, and the best place to learn them is on data whose right answers are already known. Every
stage of notebook 04 checks itself against the pandas build or against `esci/metrics.py`, so a wrong join or a wrong
regex shows up as a number that does not match, not as a vague feeling that the result looks plausible.

## 3. The shape of the code

`esci/spark.py` has one function per stage, each taking a `SparkSession` and a lake path and returning a plain
dictionary of results. No stage holds state; each reads Delta tables the previous one wrote. `run_pipeline()`
runs the requested stages in order and writes `report.json` after every one, so a run that dies in stage five
still leaves stages one to four behind, and a later call with a subset of stages keeps the earlier sections of the
report. The notebook never computes anything itself: it launches the Modal function once per run (or reads the
saved report if it exists) and renders the dictionaries as tables.

The six stages and the Spark idea each one teaches:

| Stage | What it does | Spark idea |
|---|---|---|
| `bronze` | Rewrites the two raw Parquet files as Delta tables partitioned by locale | Reading Parquet, `partitionBy` on write, how many files Spark produces and why |
| `silver` | The benchmark slice: queries with an Exact product, the official split made product-disjoint, a dev subset, passage text per product | Semi and anti joins, window functions, string functions in SQL, a pandas UDF |
| `gold` | One row per query with judged / Exact / Substitute / Complement / Irrelevant counts | `groupBy` with conditional sums, `pivot`, approximate percentiles |
| `plans` | Captures `explain()` for the two joins that matter and times them with and without broadcasting | Reading a physical plan, broadcast vs sort-merge join, partition pruning |
| `delta` | MERGE, time travel, schema enforcement, RESTORE, OPTIMIZE, VACUUM on a copy of the US judgements | The transaction log and what it buys |
| `score` | recall@k, hit@k and nDCG@k of the saved rankings from notebooks 01 and 02 | Joins plus window functions reproducing a Python evaluator to the fourth decimal |

## 4. The split as joins

Notebook 01 did the product-disjoint split with Python sets: collect every product in any test query's list,
drop training rows whose product is in that set. In Spark the same logic is three joins, because a set-membership
test over millions of rows *is* a join:

```python
has_exact  = raw.filter(F.col("esci_label") == "E").select("product_locale", "query_id").distinct()
ex         = raw.join(has_exact, ["product_locale", "query_id"], "left_semi")          # keep queries that have an Exact product
test       = ex.filter(F.col("split") == "test")
train      = ex.filter(F.col("split") == "train")
test_prods = test.select("product_locale", "product_id").distinct()
train      = train.join(test_prods, ["product_locale", "product_id"], "left_anti")     # drop every training row whose product is in a test list
still      = train.filter(F.col("esci_label") == "E").select("product_locale", "query_id").distinct()
train      = train.join(still, ["product_locale", "query_id"], "left_semi")            # drop training queries that lost their last Exact product
```

A **semi join** keeps the left rows that have a match on the right and returns only left columns; an **anti
join** keeps the left rows that have no match. They are the Spark spelling of `isin()` and `~isin()`. The keys
always include the locale, because a `product_id` or `query_id` is only guaranteed unique within a locale.

The dev subset is 500 random training queries per locale. Spark's `rand(seed)` is reproducible but it is not
numpy's `default_rng(seed)`, so the dev *membership* differs from notebook 01 while every total matches. The
honest way to verify a random step is to compare the counts, which the notebook does, or to persist the drawn
ids once and read them back, which is what the training code does with `dev_corpus_ids.json`.

One count needed reconciling. Notebook 01 reported 338 training queries dropped after the overlap removal; the
Spark build reports 349. Both are right: pandas counted queries that still had rows but lost their last Exact
product, and 11 further queries lost *every* row, so they never reached that count. 20,888 official training
queries minus 349 is 20,539 kept, in both builds.

## 5. The passage text, twice

`esci.data.product_text()` builds what the embedding model sees for a product: the whitespace-collapsed title, a
"Brand: …, Color: …" line when either exists, the bullet points, joined with newlines and cut at 1,200
characters. In Spark that becomes one column expression that runs inside the JVM:

```python
def clean(col):
    return F.regexp_replace(F.regexp_replace(F.col(col), r"(?U)^\s+|\s+$", ""), r"(?U)\s+", " ")

title   = F.coalesce(clean("product_title"), F.lit(""))
brand, color, bullets = clean("product_brand"), clean("product_color"), clean("product_bullet_point")
meta    = F.concat_ws(", ", F.when(F.length(brand) > 0, F.concat(F.lit("Brand: "), brand)),
                            F.when(F.length(color) > 0, F.concat(F.lit("Color: "), color)))
text    = F.concat_ws("\n", title, F.when(F.length(meta) > 0, meta), F.when(F.length(bullets) > 0, bullets))
passage = F.substring(text, 1, 1200)
```

Two details carry the whole translation. `concat_ws` skips nulls, and a `when` without an `otherwise` yields null,
so an absent brand line simply disappears, which is what the Python `if meta:` did. And `(?U)` switches Java's
regex engine to Unicode character classes: without it `\s` matches ASCII whitespace only, while Python's
`str.split()` treats the ideographic space U+3000, common in the Japanese data, as whitespace. The first draft
had no `(?U)`; the check below is what would have caught it.

The check is the same Python function, run inside Spark as a **pandas UDF** (User Defined Function). Spark hands
each partition to a Python worker as Arrow batches, the function receives a `pandas.Series` per column and returns
one, and Spark gets a column back. It is the slow path, because every row crosses the JVM / Python boundary,
and it exists for exactly this kind of job: when the logic already exists in Python and you want it as an oracle.
The result was identical for every product in all three locales (482,105 + 167,761 + 233,850), and byte-identical
to the `corpus.parquet` that notebook 01 wrote for the US.

## 6. Reading the plans

`df.explain("formatted")` prints the physical plan Spark chose. The two joins in the `plans` stage were picked to
show the two join algorithms, and one of them did not behave as expected, which is the more useful lesson.

**judgements ⋈ products.** 980k judgement rows against 1.8M product rows carrying the text. The expectation was a
sort-merge join: both sides shuffled by the key, sorted, merged. Spark chose a `BroadcastHashJoin` with
`BuildLeft`, the judgements side. After column pruning (three of six columns) Delta's file statistics estimated
that side under the 10 MB `autoBroadcastJoinThreshold` (the whole table is 14.7 MB on disk), so Spark copied the
judgements to every task and streamed the 1.1 GB products table through once, with no shuffle of it at all. The
rule is not "broadcast the small table"; it is "broadcast whichever side the optimiser estimates will fit",
and the estimate comes from file statistics after pruning. The only `Exchange` on the products side in that
plan is the `dropDuplicates`, which needs its own shuffle to bring equal keys together.

**train anti-join test products.** The right side, a distinct set of (locale, product_id), is also small enough to
broadcast. Setting the threshold to -1 forces the sort-merge plan for the same query, and the plan then shows
`Exchange` on both sides followed by `Sort` and `SortMergeJoin LeftAnti`. On this data the timing difference is
small (1.3 s against 1.5 s); it grows with the size of the big side, because the shuffle writes and reads every
row.

**Partition pruning.** A filter on `product_locale` shows up in the scan as a `PartitionFilters` entry, so Spark
opens only that locale's files. That is why every table is partitioned by locale: the three locales are the
natural unit of work, and most queries want one of them.

Two settings matter in local mode. `spark.sql.shuffle.partitions` is 32 instead of the default 200; the default
is tuned for clusters and on 8 cores it only adds scheduling overhead. And the master is `local[8]`, not
`local[*]`: a Modal container reports the host's 24 CPUs, but it is billed and throttled for the 8 it requested,
and `local[*]` would start 24 task slots on 8 cores' worth of CPU.

## 7. What Delta Lake changes

A Parquet table is a directory of immutable files, and that is the whole problem. A writer that dies half-way
leaves a half-written table; a row cannot be updated without rewriting its file and nothing records which files
are current; a job can append a file with a different column type; an overwritten version is gone.

Delta keeps a transaction log in `_delta_log/`, one JSON file per commit, each listing the files added and
removed and the schema. A reader starts from the log, not from a directory listing, so it only sees complete
commits. The `delta` stage exercises what that buys, on a copy of the US judgements so the real benchmark
table is never touched:

| Operation | What happened | Why it matters for a benchmark |
|---|---|---|
| `MERGE` | A simulated re-judging batch: 1,000 Substitute test pairs relabelled Exact, 500 new judgements inserted, in one atomic commit (2.9 s) | This is how corrected labels would arrive; one statement handles updates and inserts, and readers never see it half-applied |
| time travel (`versionAsOf 0`) | The table before the MERGE still reads 79,708 Exact test pairs | A frozen test set can be pinned to a version number, which is stronger than "the file as of last Tuesday" |
| schema enforcement | Appending `esci_label` as an integer was refused | The error happens at write time, in the job that is wrong, not at read time in someone else's |
| `RESTORE` | Version 0 is back as version 2; the history keeps versions 0 to 2 | Undo without a backup copy |
| `OPTIMIZE` + `VACUUM` | 14 files became 15 (the compacted file is added first) and then 1 after VACUUM removed what no live version referenced | MERGE produces small files; compaction keeps reads fast; VACUUM is what actually frees space |

`VACUUM` with 0 hours' retention is a demo setting (`retentionDurationCheck.enabled = false`); production keeps
the 7-day default so a reader in the middle of a query never loses a file.

The tables are laid out in the usual medallion way: *bronze* is the raw files as shipped, *silver* is the
cleaned and joined benchmark, *gold* is aggregates. The names are a convention, not a Delta feature; what they
encode is that each layer is reproducible from the one below it.

## 8. Getting Spark and Delta onto Modal

This was the part with no existing example in the repo. Three things had to be true inside the container:

1. **A JVM.** `apt_install("openjdk-17-jre-headless")`. Spark 4 needs Java 17 or 21.
2. **PySpark**, from pip. The wheel is 450 MB because it contains Spark itself.
3. **The Delta JARs on Spark's classpath, without a download at run time.** The `delta-spark` pip package does not
   contain them; `configure_spark_with_delta_pip()` normally makes the first session fetch them from Maven through
   Ivy into `~/.ivy2`. On a laptop that happens once; on Modal every container is fresh. So the image build
   starts a throwaway Delta session, lets Ivy resolve the JARs, and copies them into PySpark's own `jars/`
   directory. After that, a session needs only the two Delta config lines and no network.

The copy step had a trap that cost two failed runs. Ivy resolves Delta's *transitive* dependencies too: Jackson
2.13, Hadoop, Parquet, log4j, which Spark 4.2 already ships at newer versions. Copying all of them put two versions
of Jackson into `jars/`. The JVM loads whichever the directory listing returns first, and that order differed
between containers: the first smoke run passed and the second failed with
`NoSuchMethodError: JsonFactory.setStreamReadConstraints`. The fix is to copy only the `io.delta_*` and
`io.unitycatalog_*` artifacts. If a Spark job works on one machine and fails on another with a
`NoSuchMethodError`, a duplicated JAR is the first thing to suspect.

The Delta tables live in the Modal Volume `esci-data`, which the container sees as an ordinary file system. Delta
needs atomic renames for its log, which the Volume provides; the one extra step is committing the Volume after
each stage so the files persist.

## 9. What verified what

| Spark output | Checked against | Result |
|---|---|---|
| split statistics, US | `data/processed/us-small/stats.json` from notebook 01 | 18 of 18 identical |
| passage text, US | `corpus.parquet` from notebook 01 | 482,105 of 482,105 identical |
| passage text, all locales | `esci.data.product_text` in a pandas UDF | identical on every product |
| recall@k, hit@k, nDCG@k of three saved rankings | the metrics in their `report.json`, computed by `esci/metrics.py` | max difference 0.0000 |

The last row is the one to remember. A metric implementation written twice, in Python over dictionaries and in
Spark over joins and windows, agreeing to four decimals on 8,955 queries, is strong evidence that both are right.

## 10. What was deliberately left out

- **Spark SQL strings and RDDs** (section 1).
- **Cluster mode.** Local mode on one container teaches everything except network shuffle and executor
  failure. The next step, agreed, is a Spark Connect server in a Modal container driven from the laptop with the
  thin `pyspark-client` package, then Databricks Free Edition for the same tables under Unity Catalog.
- **MLlib.** The models in this project are PyTorch models trained with `sentence-transformers` on a GPU; Spark
  is the data layer, not the training engine.
- **Streaming.** The judgements arrive as a batch. The MERGE in section 7 is the batch form of what a streaming
  `foreachBatch` would do.
