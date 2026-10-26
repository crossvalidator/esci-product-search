"""The ESCI product-search library: data slicing, metrics, bootstrap, the training CLIs, BM25 and the Spark stages.

`import esci` gives `esci.data`, `esci.metrics` and `esci.bootstrap`; the heavier modules (`esci.embed_train`,
`esci.rerank_train`, `esci.baselines`, `esci.spark`) are imported explicitly where they are needed.
"""

from esci import bootstrap, data, metrics  # noqa: F401
