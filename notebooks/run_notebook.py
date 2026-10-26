"""Execute a notebook cell by cell, writing the file and a timestamped progress log after every cell.

    uv run python notebooks/run_notebook.py 01-data-and-baselines.ipynb --log logs/nb01.log
    uv run python notebooks/run_notebook.py 01-data-and-baselines.ipynb --log logs/nb01.log --until 12   # cells 0..11 only
    uv run python notebooks/run_notebook.py 01-data-and-baselines.ipynb --log logs/nb01.log --start 12   # resume

Unlike `nbclient.execute()` (which writes outputs only at the end), this leaves a readable notebook and
log behind at every step, so a long run can be watched: `tail -f logs/nb01.log`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("notebook")
    p.add_argument("--log", default=None)
    p.add_argument("--kernel", default="esci-product-search")
    p.add_argument("--start", type=int, default=0, help="first cell index to execute")
    p.add_argument("--until", type=int, default=None, help="stop before this cell index")
    p.add_argument("--timeout", type=int, default=4 * 3600, help="per-cell timeout in seconds")
    a = p.parse_args()

    path = Path(a.notebook)
    log = Path(a.log) if a.log else Path("logs") / (path.stem + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)

    def say(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(log, "a") as fh:
            fh.write(line + "\n")

    nb = nbformat.read(path, as_version=4)
    until = a.until if a.until is not None else len(nb.cells)
    client = NotebookClient(nb, timeout=a.timeout, kernel_name=a.kernel, resources={"metadata": {"path": str(path.parent or ".")}})
    say(f"executing {path} cells {a.start}..{until - 1} on kernel {a.kernel}")
    t_all = time.time()
    with client.setup_kernel():
        for i, cell in enumerate(nb.cells):
            if i < a.start or i >= until or cell.cell_type != "code":
                continue
            first = next((l for l in cell.source.strip().splitlines() if l.strip()), "")[:80]
            say(f"cell {i:2d} start | {first}")
            t0 = time.time()
            try:
                client.execute_cell(cell, i)
            except Exception as e:  # noqa: BLE001
                say(f"cell {i:2d} FAILED after {time.time() - t0:.0f}s: {type(e).__name__}: {str(e).strip().splitlines()[-1][:200]}")
                nbformat.write(nb, path)
                raise
            nbformat.write(nb, path)
            say(f"cell {i:2d} done in {time.time() - t0:.0f}s")
    say(f"finished {path} in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
