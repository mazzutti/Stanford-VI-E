# Dead Code Fixes - Line-by-Line Reference

## Quick Navigation

- [Unused Imports to Remove](#unused-imports-to-remove)
- [Unused Variables to Fix](#unused-variables-to-fix)
- [Files by Complexity](#files-by-complexity)

---

## UNUSED IMPORTS TO REMOVE

### src/analysis/processor_mixins.py

**Line 33**: Remove `Generic`

```python
# BEFORE:
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar, Union, wraps

# AFTER:
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, wraps
```

---

### src/analysis/types/base.py

**Line 14**: Remove `Enum`  
**Line 16**: Remove `Callable`

```python
# BEFORE:
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional, Type

# AFTER:
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Type
```

---

### src/analysis/pipelines/factory.py

**Line 6**: Remove `Dict`  
**Line 7**: Remove `ABC`

```python
# BEFORE:
from typing import Dict, Optional
from abc import ABC, abstractmethod

# AFTER:
from typing import Optional
from abc import abstractmethod
```

---

### src/analysis/processors/registry.py

**Line 18**: Remove `Type`  
**Line 27**: Remove `ABC, abstractmethod`

```python
# BEFORE:
from typing import Any, Dict, List, Optional, Protocol, Type

from abc import ABC, abstractmethod

# AFTER:
from typing import Any, Dict, List, Optional, Protocol

from abc import abstractmethod
```

---

### src/analysis/processors/validators.py

**Line 5**: Remove `Tuple`

```python
# BEFORE:
from typing import Any, Callable, Dict, List, Optional, Tuple

# AFTER:
from typing import Any, Callable, Dict, List, Optional
```

---

### src/analysis/facies/config.py

**Line 9**: Remove `field`

```python
# BEFORE:
from dataclasses import dataclass, field

# AFTER:
from dataclasses import dataclass
```

---

### src/analysis/facies/processor_setup.py

**Line 11**: Remove `Callable, Any`

```python
# BEFORE:
from typing import Any, Callable, Dict, List, Optional, Sequence

# AFTER:
from typing import Dict, List, Optional, Sequence
```

---

### src/analysis/facies/stages.py

**Line 16**: Remove `Optional`

```python
# BEFORE:
from typing import Any, Dict, List, Optional

# AFTER:
from typing import Any, Dict, List
```

---

### src/analysis/factories/validators.py

**Line 11**: Remove `Optional`

```python
# BEFORE:
from typing import Any, Dict, Optional, Tuple

# AFTER:
from typing import Any, Dict, Tuple
```

---

### src/analysis/rock_physics/analyzer.py

**Line 334**: Remove `PlotConfig`

```python
# BEFORE:
from src.plotting.helpers.config import PlotConfig

# AFTER:
# (Remove this import completely)
```

---

### src/analysis/validators.py

**Line 675**: Remove `ValidationResult`

```python
# BEFORE:
from src.analysis.processors.config import ValidationResult

# AFTER:
# (Remove this import, check if file should be updated)
```

---

### src/__main__.py

**Line 649**: Keep (if used)  
**Line 679**: Remove `SlicePlotter`  
**Line 706**: Remove `RockPhysicsPlotter`  
**Line 876**: Remove duplicate `PlotlyPlotter`

```python
# BEFORE (Line 649):
from src.plotting import PlotlyPlotter

# BEFORE (Line 679):
from src.plotting import SlicePlotter

# BEFORE (Line 706):
from src.plotting import RockPhysicsPlotter

# BEFORE (Line 876):
from src.plotting import PlotlyPlotter

# AFTER:
# Keep line 649, remove lines 679, 706, 876
from src.plotting import PlotlyPlotter
```

---

## UNUSED VARIABLES TO FIX

### src/__main__.py

**Line 347**: Replace loop variables with `_`

```python
# BEFORE:
for ni, nj, nz in shape_iterator:
    # Process but ni, nj, nz never used

# AFTER:
for _, _, _ in shape_iterator:
    # Or if ni is used:
    for ni, _, _ in shape_iterator:
```

**Line 361**: Replace `dt` with `_`

```python
# BEFORE:
dt = grid_spec.dt

# AFTER:
_ = grid_spec.dt
# OR remove the line if truly unnecessary
```

**Line 389**: Replace `angle_gathers, full_stack_avo` with `_`

```python
# BEFORE:
angle_gathers, full_stack_avo = get_gathers()

# AFTER:
_, _ = get_gathers()
```

**Line 1056**: Replace `DATA_PATH, FILE_MAP` with `_`

```python
# BEFORE:
DATA_PATH, FILE_MAP = unpack_config()

# AFTER:
_, _ = unpack_config()
```

---

### src/plotting/slice_plotter.py

**Line 56**: Replace unused indices with `_`

```python
# BEFORE:
for idx_i, idx_j, idx_k in enumerate_indices():
    use_idx_i(idx_i)

# AFTER:
for idx_i, _, _ in enumerate_indices():
    use_idx_i(idx_i)
```

**Line 93**: Replace unused indices with `_`

```python
# BEFORE:
for idx_i, idx_k in enumerate_indices():
    use_idx_i(idx_i)

# AFTER:
for idx_i, _ in enumerate_indices():
    use_idx_i(idx_i)
```

**Line 130**: Replace unused indices with `_`

```python
# BEFORE:
for idx_i, idx_j in enumerate_indices():
    use_idx_i(idx_i)

# AFTER:
for idx_i, _ in enumerate_indices():
    use_idx_i(idx_i)
```

---

### src/plotting/overlay_plotter.py

**Line 75**: Replace unused dimensions with `_`

```python
# BEFORE:
for nj, nk in shape_iterator:
    # Never use nj, nk

# AFTER:
for _, _ in shape_iterator:
    # (process)
```

---

### src/signal/signal.py

**Line 94**: Replace `nk` with `_`

```python
# BEFORE:
for nk in range(num_samples):
    # Process but nk not used

# AFTER:
for _ in range(num_samples):
    # Process
```

---

### src/modeling/processors.py

**Line 110**: Replace `nz` with `_`

```python
# BEFORE:
for nz in z_range:
    # nz not used in loop body

# AFTER:
for _ in z_range:
    # Process
```

---

### src/modeling/resampler.py

**Line 53**: Replace `dt` assignment with `_`

```python
# BEFORE:
dt = extracted_value

# AFTER:
_ = extracted_value
# OR remove if truly unnecessary
```

---

### src/io/disk_cache.py

**Line 246**: Replace `k` with `_`

```python
# BEFORE:
for k in range(cache_size):
    # k not used

# AFTER:
for _ in range(cache_size):
    # Process
```

---

### src/analysis/facies/stages.py

**Line 244**: Remove or comment `analyzer` assignment

```python
# BEFORE:
analyzer = create_analyzer()
# (never used)

# AFTER:
# Option 1 - Remove entirely if truly dead code
# Option 2 - If intentional:
_ = create_analyzer()
```

---

### src/analysis/factories/builder.py

**Line 980**: Check context of `proc_type`

```python
# BEFORE:
for proc_type in processor_list:
    # (process without using proc_type)

# AFTER:
for _ in processor_list:
    # (process)
```

---

### src/analysis/pipelines/orchestrator.py

**Line 391**: Check context of `stage_name`

```python
# BEFORE:
for stage_name in stages:
    # (process without using stage_name)

# AFTER:
for _ in stages:
    # (process)
```

---

### src/analysis/rock_physics/analyzer.py

**Line 503**: Remove or comment `plotter` assignment

```python
# BEFORE:
plotter = PlotterFactory.create()
# (never used)

# AFTER:
# Remove if truly dead code
_ = PlotterFactory.create()
```

---

## FILES BY COMPLEXITY

### TIER 1: Very Low Risk (17 instances of unused loop variables)

These are simple replacements of loop variables with `_`:

1. ✓ `src/plotting/slice_plotter.py` (5 vars)
2. ✓ `src/plotting/overlay_plotter.py` (2 vars)
3. ✓ `src/signal/signal.py` (1 var)
4. ✓ `src/modeling/resampler.py` (1 var)
5. ✓ `src/modeling/processors.py` (1 var)
6. ✓ `src/io/disk_cache.py` (1 var)
7. ✓ `src/__main__.py` (6 vars from unpacking)

**Effort**: ~15 minutes  
**Risk**: VERY LOW  
**Testing**: Quick smoke test

---

### TIER 2: Low Risk (3 imports + 2 standalone vars in main)

Files in `src/__main__.py`:

1. ✓ Remove 3 unused imports
2. ✓ Handle 2 standalone variable assignments

**Effort**: ~10 minutes  
**Risk**: LOW  
**Testing**: Run module imports

---

### TIER 3: Medium Risk (16 imports + 4 variables in analysis/)

Complex analysis module cleanup:

1. `src/analysis/processor_mixins.py` (1 import)
2. `src/analysis/types/base.py` (2 imports)
3. `src/analysis/pipelines/factory.py` (2 imports)
4. `src/analysis/processors/registry.py` (3 imports)
5. `src/analysis/processors/validators.py` (1 import)
6. `src/analysis/facies/config.py` (1 import + 1 var)
7. `src/analysis/facies/processor_setup.py` (2 imports)
8. `src/analysis/facies/stages.py` (1 import + 1 var)
9. `src/analysis/factories/validators.py` (1 import)
10. `src/analysis/factories/builder.py` (1 var)
11. `src/analysis/pipelines/orchestrator.py` (1 var)
12. `src/analysis/rock_physics/analyzer.py` (1 import + 1 var)
13. `src/analysis/validators.py` (1 import)

**Effort**: ~25 minutes  
**Risk**: MEDIUM  
**Testing**: Full test suite after each file

---

## EXECUTION CHECKLIST

### Before Starting
- [ ] Create clean branch: `git checkout -b fix/dead-code-cleanup`
- [ ] Verify all tests pass: `pytest -x`

### Phase 1: Loop Variables (15 min)
- [ ] Fix `src/plotting/slice_plotter.py` (5 instances)
- [ ] Fix `src/plotting/overlay_plotter.py` (2 instances)
- [ ] Fix `src/signal/signal.py` (1 instance)
- [ ] Fix `src/modeling/resampler.py` (1 instance)
- [ ] Fix `src/modeling/processors.py` (1 instance)
- [ ] Fix `src/io/disk_cache.py` (1 instance)
- [ ] Fix `src/__main__.py` (6 instances)
- [ ] Run tests: `pytest -x`
- [ ] Commit: `git commit -m "fix: replace unused loop variables with underscore"`

### Phase 2: Main Module (10 min)
- [ ] Remove 3 unused imports from `src/__main__.py`
- [ ] Handle 2 standalone variable assignments
- [ ] Run tests: `pytest -x`
- [ ] Commit: `git commit -m "fix: remove unused imports from __main__.py"`

### Phase 3: Analysis Module (25 min)
- [ ] Process each file from list above
- [ ] Run tests after 2-3 files
- [ ] Final commit: `git commit -m "fix: remove unused imports from analysis module"`

### Final Verification
- [ ] Run full test suite: `pytest -xvs`
- [ ] Check lint: `pylint --disable=all --enable=unused-import,unused-variable src/`
- [ ] Review diff: `git diff`
- [ ] Create pull request

---

**Total Estimated Time**: 50 minutes implementation + 10 minutes verification = 60 minutes

**Last Updated**: November 6, 2025
