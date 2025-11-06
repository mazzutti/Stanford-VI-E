# Dead Code Summary - Full src/ Folder (Quick Reference)

## Overall Statistics

```
Total Dead Code Issues:        43
├── Unused Imports:            21
├── Unused Variables:          22
└── Status:                    Analysis Only (Not Yet Fixed)

Files with Issues:             20 files
Modules Affected:              6 (analysis, main, plotting, modeling, io, signal)
```

---

## Unused Imports by File

| File | Count | Details |
|------|-------|---------|
| `src/analysis/processor_mixins.py` | 1 | Generic |
| `src/analysis/types/base.py` | 2 | Enum, Callable |
| `src/analysis/pipelines/factory.py` | 2 | Dict, ABC |
| `src/analysis/processors/registry.py` | 3 | Type, ABC, abstractmethod |
| `src/analysis/processors/validators.py` | 1 | Tuple |
| `src/analysis/facies/config.py` | 1 | field |
| `src/analysis/facies/processor_setup.py` | 2 | Callable, Any |
| `src/analysis/facies/stages.py` | 1 | Optional |
| `src/analysis/factories/validators.py` | 1 | Optional |
| `src/analysis/rock_physics/analyzer.py` | 1 | PlotConfig |
| `src/analysis/validators.py` | 1 | ValidationResult |
| `src/__main__.py` | 4 | PlotlyPlotter (×2), SlicePlotter, RockPhysicsPlotter |
| **TOTAL** | **21** | |

---

## Unused Variables by File

| File | Count | Details |
|------|-------|---------|
| `src/__main__.py` | 8 | ni, nj, nz, dt, angle_gathers, full_stack_avo, DATA_PATH, FILE_MAP |
| `src/plotting/slice_plotter.py` | 5 | idx_i, idx_j, idx_k (×2 each) |
| `src/plotting/overlay_plotter.py` | 2 | nj, nk |
| `src/analysis/facies/stages.py` | 1 | analyzer |
| `src/analysis/factories/builder.py` | 1 | proc_type |
| `src/analysis/pipelines/orchestrator.py` | 1 | stage_name |
| `src/analysis/rock_physics/analyzer.py` | 1 | plotter |
| `src/modeling/processors.py` | 1 | nz |
| `src/modeling/resampler.py` | 1 | dt |
| `src/io/disk_cache.py` | 1 | k |
| `src/signal/signal.py` | 1 | nk |
| **TOTAL** | **22** | |

---

## Cleanup Impact

### Module-by-Module Breakdown

| Module | Imports | Variables | Total | Complexity |
|--------|---------|-----------|-------|------------|
| `src/analysis/` | 16 | 4 | 20 | HIGH |
| `src/__main__.py` | 3 | 8 | 11 | MEDIUM |
| `src/plotting/` | 0 | 8 | 8 | LOW |
| `src/modeling/` | 0 | 2 | 2 | LOW |
| `src/io/` | 0 | 1 | 1 | LOW |
| `src/signal/` | 0 | 1 | 1 | LOW |

---

## Recommended Cleanup Order

### Phase 1: Quick Wins (30 min) - 17 issues
```
✓ src/plotting/slice_plotter.py       (5 variables)
✓ src/plotting/overlay_plotter.py     (2 variables)
✓ src/signal/signal.py                (1 variable)
✓ src/modeling/resampler.py           (1 variable)
✓ src/modeling/processors.py          (1 variable)
✓ src/io/disk_cache.py                (1 variable)
✓ src/__main__.py (loop vars)         (6 variables)
Total: 17 unused loop variables → Replace with '_'
```

### Phase 2: Main Module Cleanup (10 min) - 5 issues
```
✓ src/__main__.py imports             (3 imports)
✓ src/__main__.py variables           (2 variables)
```

### Phase 3: Analysis Module Cleanup (25 min) - 20 issues
```
✓ src/analysis/ files                 (16 imports + 4 variables)
```

---

## Type of Dead Code Patterns

### Pattern 1: Unused Loop Variables (17 instances)
```python
# Pattern: Loop unpacking with unused values
for idx_i, idx_j, idx_k in iterator:
    use(idx_i)  # idx_j, idx_k not used

# Fix: Replace with underscore
for idx_i, _, _ in iterator:
    use(idx_i)
```

### Pattern 2: Duplicate Imports (1 instance)
```python
# Line 649
from src.plotting import PlotlyPlotter

# Line 876
from src.plotting import PlotlyPlotter  # Duplicate!

# Fix: Keep one, remove other
```

### Pattern 3: Unused Type Imports (8 instances)
```python
# Example from src/analysis/processor_mixins.py:33
from typing import Generic  # Imported but never used

# Fix: Remove the import
```

### Pattern 4: Unused Variable Assignments (4 instances)
```python
# Example from src/analysis/rock_physics/analyzer.py:503
plotter = SomeFactory()  # Created but never used

# Fix: Remove if truly dead code, or add comment if intentional
```

---

## Risk Analysis

| Risk Level | Count | Examples | Mitigation |
|-----------|-------|----------|-----------|
| **VERY LOW** | 17 | Loop variable replacements | Simply replace with `_` |
| **LOW** | 4 | Unused variable assignments | Review context, confirm dead code |
| **LOW** | 3 | `__main__.py` imports | Remove imports, verify no runtime impact |
| **MEDIUM** | 16 | `src/analysis/` imports | Complex module, test thoroughly |

**Overall Risk**: LOW-MEDIUM with proper testing

---

## Before & After Comparison

### Current State
- Unused imports: 21
- Unused variables: 22
- Code quality score: 9.98/10 (per pylint)

### After Cleanup (Expected)
- Unused imports: 0
- Unused variables: 0
- Code quality score: 10.0/10

---

## Related Documents

- `DEAD_CODE_ANALYSIS_FULL_SRC.md` - Detailed analysis with implementation details
- `CLEANUP_SESSION_SUMMARY.md` - Previous successful cleanup of src/processing/
- `COMPREHENSIVE_DEAD_CODE_ANALYSIS.md` - Earlier comprehensive analysis

---

## Command Reference

```bash
# Check for unused imports and variables
pylint --disable=all --enable=unused-import,unused-variable src/

# Run tests after cleanup
pytest -xvs

# View specific file issues
pylint --disable=all --enable=unused-import,unused-variable src/plotting/slice_plotter.py
```

---

**Last Updated**: November 6, 2025  
**Status**: Ready for implementation  
**Estimated Total Time**: 75 minutes
