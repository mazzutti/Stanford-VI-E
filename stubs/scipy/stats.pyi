from typing import Any, Tuple

# Conservative stubs for scipy.stats used in repo

def pearsonr(a: Any, b: Any) -> Tuple[float, float]: ...

def spearmanr(a: Any, b: Any) -> Tuple[float, float]: ...
