"""The ESCI data pipeline again, in PySpark with Delta Lake, for all three locales (us, es, jp).

Notebook 01 built the US-small benchmark with pandas in 23 seconds. 1.1 GB of Parquet is not big data and
pandas is the right tool for it; this module exists to learn Spark's API, execution model and Delta Lake on
data whose right answers are already known (`data/processed/us-small/stats.json`, `esci/metrics.py`). Every
step verifies itself against those answers.

It runs inside one Modal CPU container in Spark local mode (one JVM, one task slot per core); nothing here
starts Spark on the laptop. `jobs/modal_spark.py` builds the image (Java 17, PySpark, the Delta JARs baked in)
and calls `run_pipeline()`.

Lakehouse layout (Delta tables, one directory each, under `<lake>/`):

    bronze_examples     raw judgements, as shipped, partitioned by locale
    bronze_products     raw products, as shipped, partitioned by locale
    silver_judgements   one locale+version slice: queries with an Exact product, official split made
                        product-disjoint, a dev subset carved from train (column `subset`)
    silver_corpus       passage text per product (title, brand/colour line, bullets; 1,200 chars), built in Spark SQL
    gold_query_stats    one row per query: judged / Exact / Substitute / Complement / Irrelevant counts, query length
    demo_judgements     a copy of the US judgements used for the MERGE / time-travel / RESTORE / OPTIMIZE demo

Stages (each returns a JSON-serialisable dict that the notebook renders):
    bronze, silver, gold, plans, delta, score
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import time
from pathlib import Path

from esci.data import PASSAGE_MAX_CHARS, product_text
from esci.metrics import GAINS

LOCALES = ("us", "es", "jp")
RAW_EXAMPLES = "shopping_queries_dataset_examples.parquet"
RAW_PRODUCTS = "shopping_queries_dataset_products.parquet"
STAGES = ("bronze", "silver", "gold", "plans", "delta", "score")


# ----------------------------------------------------------------------------------------------------------------
# Session
# ----------------------------------------------------------------------------------------------------------------

def get_spark(warehouse: str, driver_memory: str = "20g", shuffle_partitions: int = 32, cores: int | None = None, app_name: str = "esci-spark"):
    """Local-mode session with Delta enabled. The Delta JARs are already in PySpark's jars/ directory (baked into
    the Modal image at build time), so there is no Maven download here and no `configure_spark_with_delta_pip`.
    `cores` pins the task slots: a Modal container sees the host's CPUs (24 here) but is billed and throttled for
    the number it requested, and `local[*]` would oversubscribe it."""
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName(app_name)
        .master(f"local[{cores}]" if cores else "local[*]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.warehouse.dir", warehouse)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.databricks.delta.retentionDurationCheck.enabled", "false")   # lets the demo VACUUM with 0 h retention
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.extraJavaOptions", "-Dlog4j2.level=ERROR")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def session_info(spark) -> dict:
    sc = spark.sparkContext
    from esci import versions
    return {"spark_version": spark.version, "master": sc.master, "default_parallelism": sc.defaultParallelism, "host_cpus": os.cpu_count(),
            "versions": versions("pyspark", "delta-spark", "pandas", "pyarrow"),
            "shuffle_partitions": spark.conf.get("spark.sql.shuffle.partitions"),
            "driver_memory": spark.conf.get("spark.driver.memory", "default"),
            "delta_jar_on_classpath": any("delta-spark" in j for j in _jars(spark))}


def _jars(spark) -> list[str]:
    import pyspark
    return os.listdir(Path(pyspark.__file__).parent / "jars")


def _delta(spark, path: str):
    return spark.read.format("delta").load(path)


def _files(path: str) -> dict:
    """Parquet file count and bytes under a Delta table directory (the log lives in _delta_log/)."""
    n = size = 0
    for root, _, files in os.walk(path):
        if "_delta_log" in root:
            continue
        for f in files:
            if f.endswith(".parquet"):
                n += 1
                size += os.path.getsize(os.path.join(root, f))
    return {"parquet_files": n, "mb": round(size / 1e6, 1)}


def _records(df, n: int | None = None) -> list[dict]:
    rows = df.limit(n).collect() if n else df.collect()
    return [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.asDict().items()} for r in rows]


def _one_row_per_product(pr):
    """One row per (locale, product_id). `dropDuplicates` keeps whichever duplicate a task meets first, which can differ
    between runs; a window ordered by the text columns makes the choice a function of the data alone. (The ESCI products
    file has no duplicate keys, so on this data both give the same rows; this is about determinism, not correctness.)"""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    w = Window.partitionBy("product_locale", "product_id").orderBy("product_title", "product_brand", "product_color", "product_bullet_point", "product_description")
    return pr.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1).drop("_rn")


def _explain(df, mode: str = "formatted") -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        df.explain(mode=mode)
    return buf.getvalue()


# ----------------------------------------------------------------------------------------------------------------
# Bronze: raw files to Delta, as shipped
# ----------------------------------------------------------------------------------------------------------------

def bronze(spark, raw_dir: str, lake: str, log=print) -> dict:
    from pyspark.sql import functions as F

    t0 = time.time()
    rep: dict = {}
    ex = spark.read.parquet(f"{raw_dir}/{RAW_EXAMPLES}").drop("__index_level_0__")
    ex.write.format("delta").mode("overwrite").partitionBy("product_locale").save(f"{lake}/bronze_examples")
    rep["examples_seconds"] = round(time.time() - t0, 1)
    log(f"bronze_examples written in {rep['examples_seconds']}s")

    t1 = time.time()
    pr = spark.read.parquet(f"{raw_dir}/{RAW_PRODUCTS}")
    pr.write.format("delta").mode("overwrite").partitionBy("product_locale").save(f"{lake}/bronze_products")
    rep["products_seconds"] = round(time.time() - t1, 1)
    log(f"bronze_products written in {rep['products_seconds']}s")

    ex, pr = _delta(spark, f"{lake}/bronze_examples"), _delta(spark, f"{lake}/bronze_products")
    rep["examples_schema"] = [(f.name, f.dataType.simpleString()) for f in ex.schema.fields]
    rep["products_schema"] = [(f.name, f.dataType.simpleString()) for f in pr.schema.fields]
    rep["examples_per_locale"] = _records(
        ex.groupBy("product_locale").agg(
            F.count("*").alias("judgements"), F.countDistinct("query_id").alias("queries"),
            F.countDistinct("product_id").alias("products"),
            F.sum("small_version").alias("judgements_small"), F.sum("large_version").alias("judgements_large"),
            F.sum(F.when(F.col("split") == "test", 1).otherwise(0)).alias("judgements_test"),
        ).orderBy("product_locale"))
    rep["products_per_locale"] = _records(
        pr.groupBy("product_locale").agg(
            F.count("*").alias("rows"), F.countDistinct("product_id").alias("distinct_product_ids"),
            F.avg(F.length("product_title")).alias("title_chars_mean"),
            F.avg(F.when(F.col("product_bullet_point").isNotNull(), 1).otherwise(0)).alias("share_with_bullets"),
            F.avg(F.when(F.col("product_description").isNotNull(), 1).otherwise(0)).alias("share_with_description"),
        ).orderBy("product_locale"))
    rep["label_share_per_locale"] = _records(
        ex.groupBy("product_locale").pivot("esci_label", list(GAINS)).count().orderBy("product_locale"))
    rep["files"] = {"bronze_examples": _files(f"{lake}/bronze_examples"), "bronze_products": _files(f"{lake}/bronze_products")}
    rep["raw_mb"] = {RAW_EXAMPLES: round(os.path.getsize(f"{raw_dir}/{RAW_EXAMPLES}") / 1e6, 1),
                     RAW_PRODUCTS: round(os.path.getsize(f"{raw_dir}/{RAW_PRODUCTS}") / 1e6, 1)}
    rep["seconds"] = round(time.time() - t0, 1)
    return rep


# ----------------------------------------------------------------------------------------------------------------
# Silver: the benchmark slice, product-disjoint, with passage text
# ----------------------------------------------------------------------------------------------------------------

def passage_text_expr():
    """`esci.data.product_text` as Spark SQL: whitespace-collapsed title, a 'Brand: …, Color: …' line when either is
    present, the bullets, joined with newlines and cut at PASSAGE_MAX_CHARS. `(?U)` makes `\\s` match Unicode
    whitespace (ideographic spaces in the Japanese data), which is what Python's `str.split()` does."""
    from pyspark.sql import functions as F

    def clean(col):
        return F.regexp_replace(F.regexp_replace(F.col(col), r"(?U)^\s+|\s+$", ""), r"(?U)\s+", " ")

    title = F.coalesce(clean("product_title"), F.lit(""))
    brand, color, bullets = clean("product_brand"), clean("product_color"), clean("product_bullet_point")
    meta = F.concat_ws(", ", F.when(F.length(brand) > 0, F.concat(F.lit("Brand: "), brand)),
                       F.when(F.length(color) > 0, F.concat(F.lit("Color: "), color)))
    text = F.concat_ws("\n", title, F.when(F.length(meta) > 0, meta), F.when(F.length(bullets) > 0, bullets))
    return F.substring(text, 1, PASSAGE_MAX_CHARS)


def silver(spark, lake: str, locales=LOCALES, version: str = "small", dev_queries: int = 500, seed: int = 42,
           pandas_reference: str | None = None, log=print) -> dict:
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    t0 = time.time()
    rep: dict = {"locales": list(locales), "version": version, "dev_queries": dev_queries, "seed": seed, "timings": {}}
    key, pkey = ["product_locale", "query_id"], ["product_locale", "product_id"]
    cols = ["product_locale", "query_id", "query", "product_id", "esci_label"]

    raw_slice = _delta(spark, f"{lake}/bronze_examples").filter(F.col("product_locale").isin(list(locales)))
    if version == "small":
        raw_slice = raw_slice.filter(F.col("small_version") == 1)
    is_exact = F.col("esci_label") == "E"

    # 1. queries without any Exact product cannot be scored on recall: drop them (semi join on the queries that have one)
    has_exact = raw_slice.filter(is_exact).select(*key).distinct()
    ex_split = raw_slice.join(has_exact, key, "left_semi")
    test = ex_split.filter(F.col("split") == "test").select(*cols)
    train = ex_split.filter(F.col("split") == "train").select(*cols)

    # 2. product-disjoint: an anti join removes every training judgement whose product is in any test query's list
    test_products = test.select(*pkey).distinct()
    train_official = train.cache()
    train_disjoint = train_official.join(test_products, pkey, "left_anti")
    # 3. a training query that lost its last Exact product is useless for contrastive training: drop it
    still_exact = train_disjoint.filter(is_exact).select(*key).distinct()
    train_kept = train_disjoint.join(still_exact, key, "left_semi")
    # 4. dev subset: `dev_queries` random training queries per locale. Spark's rand(seed) is not numpy's
    #    default_rng(seed), so the dev *membership* differs from stats.json; the counts are what we verify.
    qids = train_kept.select(*key).distinct().withColumn("r", F.rand(seed))
    dev_ids = qids.withColumn("rn", F.row_number().over(Window.partitionBy("product_locale").orderBy("r", "query_id"))) \
                  .filter(F.col("rn") <= dev_queries).select(*key)
    dev = train_kept.join(dev_ids, key, "left_semi")
    train_final = train_kept.join(dev_ids, key, "left_anti")

    judgements = (train_final.withColumn("subset", F.lit("train"))
                  .unionByName(dev.withColumn("subset", F.lit("dev")))
                  .unionByName(test.withColumn("subset", F.lit("test"))))
    t1 = time.time()
    judgements.write.format("delta").mode("overwrite").partitionBy("product_locale").save(f"{lake}/silver_judgements")
    rep["timings"]["judgements_write"] = round(time.time() - t1, 1)
    log(f"silver_judgements written in {rep['timings']['judgements_write']}s")

    # --- per-locale statistics in the shape of stats.json, so the notebook can diff them against pandas
    t1 = time.time()
    sliced = ex_split.select(*cols)   # after the has-Exact filter

    def per_locale(df, *aggs):
        return {r["product_locale"]: {k: v for k, v in r.asDict().items() if k != "product_locale"}
                for r in df.groupBy("product_locale").agg(*aggs).collect()}

    nq, nj, npd = F.countDistinct("query_id").alias("queries"), F.count("*").alias("judgements"), F.countDistinct("product_id").alias("products")
    in_slice = per_locale(raw_slice, nq, nj, npd)
    after_exact = per_locale(sliced, nq)
    official = per_locale(train_official, nq, nj)
    leaked = per_locale(train_official.join(test_products, pkey, "left_semi"), nj, npd)
    kept = per_locale(train_kept, nq)
    jd = _delta(spark, f"{lake}/silver_judgements")
    subsets = {}
    for name in ("train", "dev", "test"):
        d = jd.filter(F.col("subset") == name)
        subsets[name] = per_locale(d, nq, nj, npd, F.sum(F.when(is_exact, 1).otherwise(0)).alias("exact_pairs"),
                                   *[F.avg(F.when(F.col("esci_label") == lab, 1.0).otherwise(0.0)).alias(f"share_{lab}") for lab in GAINS])
    seen = per_locale(jd.filter(F.col("subset") == "test").select(*pkey).distinct()
                      .join(jd.filter(F.col("subset") == "train").select(*pkey).distinct(), pkey, "left_semi"), npd)

    stats = {}
    for loc in locales:
        s = {"queries_in_slice": in_slice[loc]["queries"], "judgements_in_slice": in_slice[loc]["judgements"],
             "products_in_slice": in_slice[loc]["products"],
             "queries_dropped_no_exact": in_slice[loc]["queries"] - after_exact[loc]["queries"],
             "train_queries_official": official[loc]["queries"], "train_judgements_official": official[loc]["judgements"],
             "train_judgements_removed_product_overlap": leaked.get(loc, {}).get("judgements", 0),
             "train_products_removed_product_overlap": leaked.get(loc, {}).get("products", 0),
             "train_queries_dropped_after_overlap_removal": official[loc]["queries"] - kept[loc]["queries"],
             "test_products_seen_in_final_train": seen.get(loc, {}).get("products", 0)}
        for name in ("train", "dev", "test"):
            r = subsets[name].get(loc, {})
            s[name] = {"queries": r.get("queries", 0), "judgements": r.get("judgements", 0), "products": r.get("products", 0),
                       "exact_pairs": r.get("exact_pairs", 0),
                       "label_share": {lab: round(float(r.get(f"share_{lab}", 0.0)), 3) for lab in GAINS}}
        stats[loc] = s
    rep["stats"] = stats
    rep["timings"]["stats"] = round(time.time() - t1, 1)
    train_official.unpersist()

    # --- corpus: every product in the slice, with passage text built in Spark SQL
    t1 = time.time()
    pr = _delta(spark, f"{lake}/bronze_products").filter(F.col("product_locale").isin(list(locales)))
    dup = pr.groupBy(*pkey).count().filter(F.col("count") > 1).count()
    pr = _one_row_per_product(pr)
    slice_products = raw_slice.select(*pkey).distinct()
    corpus = (slice_products.join(pr, pkey, "left")
              .withColumn("text", passage_text_expr())
              .select("product_locale", "product_id", F.col("product_title").alias("title"), F.col("product_brand").alias("brand"),
                      F.col("product_color").alias("color"), "text", F.length("text").alias("n_chars"),
                      F.col("product_title").isNull().alias("missing_from_products_file")))
    corpus.write.format("delta").mode("overwrite").partitionBy("product_locale").save(f"{lake}/silver_corpus")
    rep["timings"]["corpus_write"] = round(time.time() - t1, 1)
    corpus = _delta(spark, f"{lake}/silver_corpus")
    rep["corpus"] = per_locale(
        corpus, F.count("*").alias("products"), F.sum(F.col("missing_from_products_file").cast("int")).alias("missing_from_products_file"),
        F.avg("n_chars").alias("text_chars_mean"), F.percentile_approx("n_chars", 0.5, 10_000).alias("text_chars_median"),
        F.avg((F.col("n_chars") >= PASSAGE_MAX_CHARS).cast("double")).alias("share_cut_at_max_chars"),
        F.avg(F.col("brand").isNotNull().cast("double")).alias("share_with_brand"))
    rep["products_file_duplicate_keys"] = dup
    log(f"silver_corpus written in {rep['timings']['corpus_write']}s; duplicate (locale, product_id) keys in products file: {dup}")

    # --- verification 1: the SQL passage text against the pandas text of notebook 01 (US-small corpus.parquet)
    if pandas_reference and os.path.exists(pandas_reference):
        t1 = time.time()
        ref = spark.read.parquet(pandas_reference).select("product_id", F.col("text").alias("pandas_text"))
        both = corpus.filter(F.col("product_locale") == "us").join(ref, "product_id", "inner")
        same = both.filter(F.col("text") == F.col("pandas_text")).count()
        total = both.count()
        diffs = both.filter(F.col("text") != F.col("pandas_text")).select(
            "product_id", F.substring("text", 1, 120).alias("spark_text"), F.substring("pandas_text", 1, 120).alias("pandas_text")).limit(5)
        rep["verify_vs_pandas_corpus"] = {"products_compared": total, "identical": same, "differ": total - same,
                                          "reference_rows": ref.count(), "examples": _records(diffs), "seconds": round(time.time() - t1, 1)}
        log(f"text verification: {same}/{total} identical to the pandas corpus")

    # --- verification 2: the same Python function, applied in a pandas UDF to every product of every locale
    t1 = time.time()
    import pandas as pd
    from pyspark.sql.functions import pandas_udf

    def _python_text(title, brand, color, bullets):
        return pd.Series([product_text(t, b, c, bp) for t, b, c, bp in zip(title, brand, color, bullets, strict=True)])
    # pandas_udf infers the Series-to-Series kind from the type hints; `from __future__ import annotations` would turn
    # them into strings it cannot read, so they are attached as real objects here.
    _python_text.__annotations__ = {"title": pd.Series, "brand": pd.Series, "color": pd.Series, "bullets": pd.Series, "return": pd.Series}
    python_text = pandas_udf("string")(_python_text)

    raw_corpus = slice_products.join(pr, pkey, "left").withColumn("sql_text", passage_text_expr()) \
        .withColumn("py_text", python_text("product_title", "product_brand", "product_color", "product_bullet_point"))
    agg = raw_corpus.groupBy("product_locale").agg(
        F.count("*").alias("products"), F.sum((F.col("sql_text") == F.col("py_text")).cast("int")).alias("identical")).orderBy("product_locale")
    rep["verify_sql_vs_pandas_udf"] = {"per_locale": _records(agg), "seconds": round(time.time() - t1, 1),
                                       "examples": _records(raw_corpus.filter(F.col("sql_text") != F.col("py_text")).select(
                                           "product_locale", "product_id", F.substring("sql_text", 1, 120).alias("sql_text"),
                                           F.substring("py_text", 1, 120).alias("py_text")).limit(5))}
    log(f"pandas UDF verification done in {rep['verify_sql_vs_pandas_udf']['seconds']}s")

    rep["files"] = {t: _files(f"{lake}/{t}") for t in ("silver_judgements", "silver_corpus")}
    rep["seconds"] = round(time.time() - t0, 1)
    return rep


# ----------------------------------------------------------------------------------------------------------------
# Gold: per-query statistics
# ----------------------------------------------------------------------------------------------------------------

def gold(spark, lake: str, log=print) -> dict:
    from pyspark.sql import functions as F

    t0 = time.time()
    jd = _delta(spark, f"{lake}/silver_judgements")
    q = jd.groupBy("product_locale", "query_id", "subset").agg(
        F.first("query").alias("query"), F.count("*").alias("n_judged"),
        *[F.sum(F.when(F.col("esci_label") == lab, 1).otherwise(0)).alias(f"n_{lab}") for lab in GAINS])
    q = q.withColumn("query_chars", F.length("query")).withColumn("query_words", F.size(F.split(F.trim("query"), r"(?U)\s+"))) \
         .withColumn("all_exact", F.col("n_E") == F.col("n_judged"))
    q.write.format("delta").mode("overwrite").partitionBy("product_locale").save(f"{lake}/gold_query_stats")
    q = _delta(spark, f"{lake}/gold_query_stats")

    rep: dict = {}
    rep["per_locale"] = _records(q.groupBy("product_locale").agg(
        F.count("*").alias("queries"), F.avg("n_judged").alias("judged_per_query_mean"),
        F.percentile_approx("n_judged", 0.5, 10_000).alias("judged_per_query_median"), F.max("n_judged").alias("judged_per_query_max"),
        F.avg("n_E").alias("exact_per_query_mean"), F.percentile_approx("n_E", 0.5, 10_000).alias("exact_per_query_median"),
        F.avg(F.col("all_exact").cast("double")).alias("share_queries_all_exact"),
        F.avg("query_words").alias("query_words_mean"), F.avg("query_chars").alias("query_chars_mean")).orderBy("product_locale"))
    rep["per_locale_subset"] = _records(q.groupBy("product_locale", "subset").agg(
        F.count("*").alias("queries"), F.sum("n_judged").alias("judgements"), F.sum("n_E").alias("exact_pairs"),
        F.avg("n_E").alias("exact_per_query_mean")).orderBy("product_locale", "subset"))
    rep["judged_histogram"] = _records(q.withColumn("bucket", F.when(F.col("n_judged") < 10, "<10").when(F.col("n_judged") < 20, "10-19")
                                                     .when(F.col("n_judged") < 40, "20-39").otherwise("40+"))
                                        .groupBy("product_locale").pivot("bucket", ["<10", "10-19", "20-39", "40+"]).count().orderBy("product_locale"))
    rep["most_judged"] = _records(q.orderBy(F.desc("n_judged")).select("product_locale", "query_id", "query", "n_judged", "n_E", "n_S", "n_C", "n_I"), 8)
    rep["files"] = {"gold_query_stats": _files(f"{lake}/gold_query_stats")}
    rep["seconds"] = round(time.time() - t0, 1)
    log(f"gold done in {rep['seconds']}s")
    return rep


# ----------------------------------------------------------------------------------------------------------------
# Plans: what Spark does for the two joins that matter
# ----------------------------------------------------------------------------------------------------------------

def plans(spark, lake: str, log=print) -> dict:
    from pyspark.sql import functions as F

    t0 = time.time()
    rep: dict = {"autoBroadcastJoinThreshold": spark.conf.get("spark.sql.autoBroadcastJoinThreshold"),
                 "adaptive_enabled": spark.conf.get("spark.sql.adaptive.enabled")}
    jd = _delta(spark, f"{lake}/silver_judgements")
    pr = _delta(spark, f"{lake}/bronze_products").dropDuplicates(["product_locale", "product_id"])
    pkey = ["product_locale", "product_id"]

    # The big one: every judgement to its product row (1.8M products with text) — too large to broadcast
    big = jd.join(pr, pkey, "inner").select("product_locale", "query_id", "product_id", "esci_label", "product_title")
    rep["big_join_plan"] = _explain(big)
    t1 = time.time()
    rep["big_join_rows"] = big.count()
    rep["big_join_seconds"] = round(time.time() - t1, 1)
    log(f"big join: {rep['big_join_rows']} rows in {rep['big_join_seconds']}s")

    # The small one: the anti join that makes training product-disjoint; the test-product set is small enough to broadcast
    test_products = jd.filter(F.col("subset") == "test").select(*pkey).distinct()
    train = jd.filter(F.col("subset") != "test")
    anti = train.join(test_products, pkey, "left_anti")
    rep["anti_join_plan"] = _explain(anti)
    t1 = time.time()
    rep["anti_join_rows"] = anti.count()
    rep["anti_join_seconds"] = round(time.time() - t1, 1)

    # The same anti join with broadcasting switched off: Spark falls back to a sort-merge join (both sides shuffled)
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
    anti_smj = train.join(test_products, pkey, "left_anti")
    rep["anti_join_no_broadcast_plan"] = _explain(anti_smj)
    t1 = time.time()
    rep["anti_join_no_broadcast_rows"] = anti_smj.count()
    rep["anti_join_no_broadcast_seconds"] = round(time.time() - t1, 1)
    spark.conf.unset("spark.sql.autoBroadcastJoinThreshold")
    log(f"anti join: broadcast {rep['anti_join_seconds']}s vs sort-merge {rep['anti_join_no_broadcast_seconds']}s")

    # Partition pruning: a filter on the partition column reads one locale's files only
    one = _delta(spark, f"{lake}/silver_judgements").filter(F.col("product_locale") == "jp")
    rep["partition_pruning_plan"] = _explain(one, "simple")
    rep["seconds"] = round(time.time() - t0, 1)
    return rep


# ----------------------------------------------------------------------------------------------------------------
# Delta: MERGE, history, time travel, schema enforcement, RESTORE, OPTIMIZE, VACUUM, on a copy of the US judgements
# ----------------------------------------------------------------------------------------------------------------

def delta_ops(spark, lake: str, seed: int = 42, log=print) -> dict:
    from delta.tables import DeltaTable
    from pyspark.sql import functions as F

    t0 = time.time()
    rep: dict = {}
    demo = f"{lake}/demo_judgements"
    src = _delta(spark, f"{lake}/silver_judgements").filter(F.col("product_locale") == "us")
    src.write.format("delta").mode("overwrite").save(demo)                     # version 0
    tbl = DeltaTable.forPath(spark, demo)

    def exact_in_test(df):
        return df.filter((F.col("subset") == "test") & (F.col("esci_label") == "E")).count()

    def history():
        return _records(tbl.history().select("version", "operation", "operationMetrics").orderBy("version"))

    rep["rows_v0"], rep["exact_test_v0"] = src.count(), exact_in_test(_delta(spark, demo))
    rep["files_v0"] = _files(demo)

    # A re-judging batch arrives: 1,000 Substitute pairs in the test set were re-labelled Exact, and 500 brand-new
    # judgements (products not in the table) were added. One MERGE applies both: update matches, insert the rest.
    t1 = time.time()
    relabel = (src.filter((F.col("subset") == "test") & (F.col("esci_label") == "S")).orderBy(F.rand(seed)).limit(1000)
               .withColumn("esci_label", F.lit("E")))
    new_rows = (src.filter(F.col("subset") == "test").orderBy(F.rand(seed + 1)).limit(500)
                .withColumn("product_id", F.concat(F.lit("NEW-"), F.col("product_id"))).withColumn("esci_label", F.lit("I")))
    updates = relabel.unionByName(new_rows)
    (tbl.alias("t").merge(updates.alias("u"), "t.product_locale = u.product_locale AND t.query_id = u.query_id AND t.product_id = u.product_id")
        .whenMatchedUpdate(set={"esci_label": "u.esci_label"})
        .whenNotMatchedInsertAll()
        .execute())                                                                # version 1
    rep["merge_seconds"] = round(time.time() - t1, 1)
    rep["rows_v1"], rep["exact_test_v1"] = _delta(spark, demo).count(), exact_in_test(_delta(spark, demo))
    log(f"MERGE in {rep['merge_seconds']}s: rows {rep['rows_v0']} -> {rep['rows_v1']}, Exact test pairs {rep['exact_test_v0']} -> {rep['exact_test_v1']}")

    # Time travel: the frozen benchmark is still there at version 0
    v0 = spark.read.format("delta").option("versionAsOf", 0).load(demo)
    rep["exact_test_versionAsOf_0"] = exact_in_test(v0)

    # Schema enforcement: an append with the wrong type is refused
    bad = spark.createDataFrame([("us", 1, "q", "B0", 1, "test")], ["product_locale", "query_id", "query", "product_id", "esci_label", "subset"])
    try:
        bad.write.format("delta").mode("append").save(demo)
        rep["schema_enforcement"] = "append with esci_label as int was ACCEPTED (unexpected)"
    except Exception as e:   # DeltaAnalysisException, a subclass of AnalysisException
        rep["schema_enforcement"] = f"refused ({type(e).__name__}): " + str(e).split("\n")[0][:300]

    # RESTORE puts version 0 back as a new version (the history keeps everything)
    tbl.restoreToVersion(0)                                                        # version 2
    rep["exact_test_after_restore"] = exact_in_test(_delta(spark, demo))

    # OPTIMIZE compacts the small files the MERGE created; VACUUM deletes files no live version needs
    before = _files(demo)
    tbl.optimize().executeCompaction()                                             # version 3
    after_opt = _files(demo)
    tbl.vacuum(0)
    after_vac = _files(demo)
    rep["files"] = {"before_optimize": before, "after_optimize": after_opt, "after_vacuum": after_vac}
    rep["rows_final"] = _delta(spark, demo).count()
    rep["history"] = history()
    rep["seconds"] = round(time.time() - t0, 1)
    log(f"delta ops done in {rep['seconds']}s; files {before['parquet_files']} -> {after_opt['parquet_files']} -> {after_vac['parquet_files']}")
    return rep


# ----------------------------------------------------------------------------------------------------------------
# Score: recall@k, hit@k and nDCG@k of saved rankings, computed with window functions
# ----------------------------------------------------------------------------------------------------------------

def score_rankings(spark, lake: str, rankings: dict[str, str], locale: str = "us", ks=(10, 20, 50), log=print) -> dict:
    """`rankings` maps a label to a rankings parquet (query_id, rank, product_id, score; rank is 1-based, 100 per query).
    Same definitions as esci.metrics.evaluate: recall@k over Exact products, hit@k, graded nDCG with unjudged = 0."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    t0 = time.time()
    jd = _delta(spark, f"{lake}/silver_judgements").filter((F.col("product_locale") == locale) & (F.col("subset") == "test"))
    gain = F.create_map(*[x for lab, g in GAINS.items() for x in (F.lit(lab), F.lit(g))])
    qrels = jd.select("query_id", "product_id", "esci_label").withColumn("gain", gain[F.col("esci_label")])
    per_query = qrels.groupBy("query_id").agg(F.sum(F.when(F.col("esci_label") == "E", 1).otherwise(0)).alias("n_exact"))
    w = Window.partitionBy("query_id").orderBy(F.desc("gain"), "product_id")
    ideal = qrels.withColumn("rn", F.row_number().over(w))
    for k in ks:
        idcg = ideal.filter(F.col("rn") <= k).groupBy("query_id").agg(F.sum(F.col("gain") / F.log2(F.col("rn") + 1)).alias(f"idcg@{k}"))
        per_query = per_query.join(idcg, "query_id", "left")
    per_query = per_query.cache()
    n_queries = per_query.count()

    rep: dict = {"locale": locale, "test_queries": n_queries, "results": {}}
    for label, path in rankings.items():
        if not os.path.exists(path):   # a fresh account has not run notebooks 01 and 02 yet: say so instead of failing the last stage
            rep["results"][label] = {"path": path, "skipped": "rankings file not found; run notebooks 01 and 02 first, or pass your own run names"}
            log(f"{label}: skipped, {path} not found")
            continue
        r = spark.read.parquet(path).select("query_id", "rank", "product_id")
        r = r.join(qrels.select("query_id", "product_id", "esci_label", "gain"), ["query_id", "product_id"], "left") \
             .withColumn("gain", F.coalesce("gain", F.lit(0.0))).withColumn("is_exact", (F.col("esci_label") == "E").cast("int")).fillna({"is_exact": 0})
        aggs = []
        for k in ks:
            in_k = F.col("rank") <= k
            aggs += [F.sum(F.when(in_k, F.col("is_exact")).otherwise(0)).alias(f"exact@{k}"),
                     F.max(F.when(in_k, F.col("is_exact")).otherwise(0)).alias(f"hit@{k}"),
                     F.sum(F.when(in_k, F.col("gain") / F.log2(F.col("rank") + 1)).otherwise(0.0)).alias(f"dcg@{k}")]
        pq = per_query.join(r.groupBy("query_id").agg(*aggs), "query_id", "left").fillna(0)
        metrics = {}
        for k in ks:
            row = pq.agg(F.avg(F.col(f"exact@{k}") / F.col("n_exact")).alias("recall"), F.avg(f"hit@{k}").alias("hit"),
                         F.avg(F.col(f"dcg@{k}") / F.col(f"idcg@{k}")).alias("ndcg")).collect()[0]
            metrics[f"recall@{k}"], metrics[f"hit@{k}"], metrics[f"ndcg@{k}"] = round(row["recall"], 4), round(row["hit"], 4), round(row["ndcg"], 4)
        rep["results"][label] = {"path": path, "metrics": metrics, "ranked_rows": r.count()}
        log(f"{label}: " + ", ".join(f"{k} {v:.4f}" for k, v in metrics.items()))
    per_query.unpersist()
    rep["seconds"] = round(time.time() - t0, 1)
    return rep


# ----------------------------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------------------------

def run_pipeline(run_name: str, raw_dir: str, lake: str, out_dir: str, stages=STAGES, locales=LOCALES, version: str = "small",
                 dev_queries: int = 500, seed: int = 42, pandas_reference: str | None = None, rankings: dict[str, str] | None = None,
                 driver_memory: str = "20g", shuffle_partitions: int = 32, cores: int | None = None, checkpoint=None, log=print) -> dict:
    """Run the requested stages in order, saving `<out_dir>/report.json` after every stage (and calling `checkpoint()`
    so a Modal Volume can be committed), so a run that dies late still leaves its earlier results behind. A later
    call with a subset of stages keeps the earlier stages' sections of an existing report."""
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spark = get_spark(warehouse=f"{lake}/_warehouse", driver_memory=driver_memory, shuffle_partitions=shuffle_partitions, cores=cores)
    # The pandas UDF in `silver` imports esci.data on the Python workers: ship the package to them as a zip.
    import shutil
    pkg = Path(__file__).resolve().parent
    spark.sparkContext.addPyFile(shutil.make_archive("/tmp/esci_pkg", "zip", root_dir=str(pkg.parent), base_dir=pkg.name))
    previous = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
    report: dict = {"run_name": run_name, "lake": lake, "stages_requested": list(stages), "locales": list(locales), "version": version,
                    "session": session_info(spark), "stages": previous.get("stages", {}), "stage_seconds": previous.get("stage_seconds", {}),
                    "previous_container_seconds": previous.get("container_seconds", 0) + previous.get("previous_container_seconds", 0)}
    log(f"Spark {report['session']['spark_version']} local mode, {report['session']['default_parallelism']} task slots")

    def save():
        report["elapsed_seconds"] = round(time.time() - t0, 1)
        (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
        if checkpoint:
            checkpoint()

    for stage in stages:
        t1 = time.time()
        log(f"--- stage {stage}")
        if stage == "bronze":
            res = bronze(spark, raw_dir, lake, log)
        elif stage == "silver":
            res = silver(spark, lake, locales, version, dev_queries, seed, pandas_reference, log)
        elif stage == "gold":
            res = gold(spark, lake, log)
        elif stage == "plans":
            res = plans(spark, lake, log)
        elif stage == "delta":
            res = delta_ops(spark, lake, seed, log)
        elif stage == "score":
            res = score_rankings(spark, lake, rankings or {}, "us", log=log) if rankings else {"skipped": "no rankings given"}
        else:
            raise ValueError(f"unknown stage {stage}")
        report["stages"][stage] = res
        report["stage_seconds"][stage] = round(time.time() - t1, 1)
        save()
    spark.stop()
    return report
