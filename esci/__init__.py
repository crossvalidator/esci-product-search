"""The ESCI product-search library: data slicing, metrics, bootstrap, the training CLIs, BM25 and the Spark stages.

`import esci` gives `esci.data`, `esci.metrics` and `esci.bootstrap`; the heavier modules (`esci.embed_train`,
`esci.rerank_train`, `esci.baselines`, `esci.spark`) are imported explicitly where they are needed.
"""

from esci import bootstrap, data, metrics  # noqa: F401


def versions(*packages: str) -> dict[str, str]:
    """Installed versions of the libraries a run depends on, written into every report.json so a result can be
    tied to the exact torch / sentence-transformers / PySpark it was produced with."""
    import importlib.metadata as meta

    out = {}
    for name in packages:
        try:
            out[name] = meta.version(name)
        except meta.PackageNotFoundError:
            out[name] = "not installed"
    return out
