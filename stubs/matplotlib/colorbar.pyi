from collections.abc import Sequence
from typing import Any

class Colorbar:
    def set_ticklabels(
        self, labels: Sequence[str], *args: Any, **kwargs: Any
    ) -> None: ...
    def set_ticks(self, ticks: Sequence[float], *args: Any, **kwargs: Any) -> None: ...
