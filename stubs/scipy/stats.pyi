from typing import Any, NamedTuple

# Conservative stubs for scipy.stats used in repo

__all__ = ["PearsonRResult", "pearsonr", "spearmanr"]

class PearsonRResult(NamedTuple):
    statistic: float
    pvalue: float

def pearsonr(a: Any, b: Any) -> PearsonRResult: ...
def spearmanr(a: Any, b: Any) -> tuple[float, float]: ...
