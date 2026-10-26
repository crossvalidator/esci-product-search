"""Shared by the Modal launchers and the notebooks: list prices, the cost estimate and `modal_once`."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

# Modal list prices (USD per hour) as of October 2026, for the cost estimate printed after each run.
# The invoice in the Modal dashboard is the measured number; this is seconds × list price.
RATES = {"A10G": 1.10, "L4": 0.80, "L40S": 1.95, "A100-80GB": 2.50, "H100": 3.95, "cpu-core": 0.047, "memory-GiB": 0.008}


def est_cost(seconds: float, gpu: str | None = None, cpu: float = 0, mem_gib: float = 0) -> float:
    hours = seconds / 3600
    return round(hours * ((RATES.get(gpu, 0) if gpu else 0) + cpu * RATES["cpu-core"] + mem_gib * RATES["memory-GiB"]), 3)


def modal_once(run_name: str, launch: Callable[[], dict], download: Callable[[str], Path] | None = None,
               out_dir: str | Path = "outputs", report_name: str = "report.json") -> dict:
    """Run a Modal job once: if `<out_dir>/<run_name>/<report_name>` is already there, read it back instead of paying
    for the job again; otherwise call `launch()`, then `download(run_name)` to bring the run's files back.
    `download` defaults to `jobs.modal_esci.download_run`; the Spark notebook passes `jobs.modal_spark.download_run`."""
    p = Path(out_dir) / run_name / report_name
    if p.exists():
        print(f"{run_name}: using the saved report at {p}")
        return json.loads(p.read_text())
    report = launch()
    if download is None:
        from jobs.modal_esci import download_run as download
    download(run_name)
    return report
