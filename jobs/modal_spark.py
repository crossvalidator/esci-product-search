"""Spark + Delta Lake on Modal for the ESCI data (notebook 04). One CPU container, Spark in local mode.

    modal run jobs/modal_spark.py::pipeline --run-name spark-smoke --locales us            # US only, verified against stats.json
    modal run jobs/modal_spark.py::pipeline --run-name spark-all --locales us,es,jp         # all three locales
    modal run jobs/modal_spark.py::pipeline --run-name spark-all --stages delta,score       # rerun two stages on an existing lake

The image installs Java 17 and PySpark, then starts a throwaway Delta session at *build* time so the Delta JARs
(normally fetched from Maven on every first session) are copied into PySpark's jars/ directory once. At run time
there is no network dependency and no `configure_spark_with_delta_pip`.

Reads the raw files from the `esci-data` Volume (downloaded by `jobs/modal_esci.py::data`), writes the Delta lake to
`esci-data/lake/<run-name>/` and the report to `esci-runs/<run-name>/report.json`. Nothing in `jobs/modal_esci.py` is
touched; this file only reuses the two Volumes by name.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # the project root, so `esci` is importable from jobs/

PYSPARK, DELTA = "4.2.0", "4.4.1"            # delta-spark 4.4.x is built for Spark 4.2; Spark 4 needs Java 17 or 21
CPU, MEMORY_MB = 8, 32_768
RATES = {"cpu-core": 0.047, "memory-GiB": 0.008}   # Modal list prices, USD per hour, October 2026

# Resolve the Delta JARs through Ivy once, copy them next to Spark's own JARs, remove the probe table. Only the
# io.delta / io.unitycatalog artifacts are copied: Ivy also resolves transitive JARs (Jackson 2.13, Hadoop, Parquet,
# log4j) that Spark 4.2 already ships at newer versions, and two versions of one library in jars/ load in
# directory order, which differs between containers (the first smoke run passed, the second failed on Jackson).
BAKE_DELTA = (
    "python -c \""
    "from pyspark.sql import SparkSession; from delta import configure_spark_with_delta_pip as c; "
    "b = SparkSession.builder.master('local[1]').config('spark.jars.ivy', '/root/.ivy2')"
    ".config('spark.sql.extensions', 'io.delta.sql.DeltaSparkSessionExtension')"
    ".config('spark.sql.catalog.spark_catalog', 'org.apache.spark.sql.delta.catalog.DeltaCatalog'); "
    "s = c(b).getOrCreate(); s.range(3).write.format('delta').mode('overwrite').save('/tmp/probe'); "
    "print('delta probe rows', s.read.format('delta').load('/tmp/probe').count()); s.stop()\" "
    "&& cp /root/.ivy2/jars/io.delta_*.jar /root/.ivy2/jars/io.unitycatalog_*.jar \"$(python -c 'import pyspark, os; print(os.path.join(os.path.dirname(pyspark.__file__), \"jars\"))')\"/ "
    "&& rm -rf /tmp/probe /root/.ivy2 && ls \"$(python -c 'import pyspark, os; print(os.path.join(os.path.dirname(pyspark.__file__), \"jars\"))')\" | grep -i delta"
)

app = modal.App("esci-product-search-spark")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("openjdk-17-jre-headless", "procps")
    .pip_install(f"pyspark[sql,pandas-on-spark]=={PYSPARK}", f"delta-spark=={DELTA}", "pandas>=2.2,<3", "pyarrow>=17", "numpy>=1.26")
    .run_commands(BAKE_DELTA)
    .env({"SPARK_LOCAL_IP": "127.0.0.1", "PYSPARK_PYTHON": "python", "PYSPARK_DRIVER_PYTHON": "python"})
    .add_local_python_source("esci")          # the library package (esci.spark, esci.data), mounted at /root/esci
)
data_vol = modal.Volume.from_name("esci-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("esci-runs", create_if_missing=True)
DATA, RUNS = "/vol/data", "/vol/runs"

# Saved rankings (esci-runs Volume) scored again in Spark, against the numbers in their report.json
DEFAULT_RANKINGS = {
    "bge-base, off the shelf": "baselines-us-small/rankings/BAAI__bge-base-en-v1.5.parquet",
    "bge-base fine-tuned (labelled)": "embed-v1-labelled/rankings/after.parquet",
    "hybrid: BM25 + fine-tuned": "embed-v1-substitutes-hybrid/rankings/hybrid(bm25+after).parquet",
}


def est_cost(seconds: float, cpu: float = CPU, mem_gib: float = MEMORY_MB / 1024) -> float:
    return round(seconds / 3600 * (cpu * RATES["cpu-core"] + mem_gib * RATES["memory-GiB"]), 3)


@app.function(image=image, cpu=CPU, memory=MEMORY_MB, timeout=2 * 3600, volumes={DATA: data_vol, RUNS: runs_vol})
def spark_pipeline(run_name: str, stages: list[str], locales: list[str], version: str = "small", dev_queries: int = 500, seed: int = 42,
                   rankings: dict[str, str] | None = None, driver_memory: str = "20g", shuffle_partitions: int = 32) -> dict:
    sys.path.insert(0, "/root")
    from esci import spark

    t0 = time.time()
    runs_vol.reload()
    out = Path(RUNS) / run_name

    def checkpoint():
        data_vol.commit()
        runs_vol.commit()

    report = spark.run_pipeline(
        run_name, raw_dir=f"{DATA}/raw", lake=f"{DATA}/lake/{run_name}", out_dir=str(out), stages=stages, locales=locales,
        version=version, dev_queries=dev_queries, seed=seed, pandas_reference=f"{DATA}/processed/us-small/corpus.parquet",
        rankings={k: f"{RUNS}/{v}" for k, v in (rankings or {}).items()} or None,
        driver_memory=driver_memory, shuffle_partitions=shuffle_partitions, cores=CPU, checkpoint=checkpoint)
    report["container_seconds"] = round(time.time() - t0, 1)
    report["cpu"], report["memory_mb"] = CPU, MEMORY_MB
    report["estimated_cost_usd"] = est_cost(report["container_seconds"] + report.get("previous_container_seconds", 0))
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    runs_vol.commit()
    return report


def download_run(run_name: str, local_root: str = "outputs") -> Path:
    """Fetch `<run>/report.json` (a few hundred KB: statistics, plans, metrics) to outputs/<run>/."""
    dest = Path(local_root) / run_name
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / "report.json", "wb") as fh:
        for chunk in runs_vol.read_file(f"{run_name}/report.json"):
            fh.write(chunk)
    return dest


def print_summary(report: dict) -> None:
    st = report["stages"]
    print(f"Spark {report['session']['spark_version']}, {report['session']['default_parallelism']} task slots, stages: "
          + ", ".join(f"{k} {v:.0f}s" for k, v in report["stage_seconds"].items()))
    if "silver" in st and "stats" in st["silver"]:
        for loc, s in st["silver"]["stats"].items():
            print(f"  {loc}: {s['queries_in_slice']:,} queries, {s['judgements_in_slice']:,} judgements, {s['products_in_slice']:,} products; "
                  f"train {s['train']['queries']:,} q / dev {s['dev']['queries']:,} q / test {s['test']['queries']:,} q; "
                  f"removed {s['train_judgements_removed_product_overlap']:,} overlapping judgements; leaked test products {s['test_products_seen_in_final_train']}")
        v = st["silver"].get("verify_vs_pandas_corpus")
        if v:
            print(f"  passage text vs pandas corpus: {v['identical']:,}/{v['products_compared']:,} identical")
    if "score" in st and "results" in st["score"]:
        for label, r in st["score"]["results"].items():
            m = r["metrics"]
            print(f"  {label:34s} recall@10 {m['recall@10']:.4f}  recall@20 {m['recall@20']:.4f}  recall@50 {m['recall@50']:.4f}  hit@20 {m['hit@20']:.4f}  ndcg@20 {m['ndcg@20']:.4f}")
    secs = report.get("container_seconds", report.get("elapsed_seconds", 0)) + report.get("previous_container_seconds", 0)
    print(f"container time {secs:.0f}s (all calls for this run) on {CPU} cores / {MEMORY_MB // 1024} GB; estimated cost at list price ${est_cost(secs):.3f}")


@app.local_entrypoint()
def pipeline(run_name: str = "spark-smoke", stages: str = ",".join(("bronze", "silver", "gold", "plans", "delta", "score")),
             locales: str = "us", version: str = "small", dev_queries: int = 500, seed: int = 42, score: bool = True) -> None:
    stage_list = [s for s in stages.split(",") if s]
    report = spark_pipeline.remote(run_name, stage_list, [l for l in locales.split(",") if l], version, dev_queries, seed,
                                   DEFAULT_RANKINGS if score else None)
    download_run(run_name)
    print_summary(report)
