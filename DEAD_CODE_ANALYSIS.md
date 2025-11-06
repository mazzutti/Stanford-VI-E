# Dead Code Analysis - src/processing Module

## Summary
Found and identified unused code in the processing module that was created but never actually used in the codebase.

## Already Removed (3 files)
✅ Deleted `src/processing/seismogram.py` - Unused `SeismoCube` class
✅ Deleted `src/processing/config.py` - Unused `ProcessingConfig` singleton
✅ Deleted `src/processing/align.py` - Unused `align_cubes()` function

## Additional Dead Code Identified

### 1. In `src/processing/services.py`

#### Unused Exports and Functions:
- `get_rock_physics_service()` - Factory function, never imported or used
- `RockPhysicsService` - Service class, never instantiated except internally
- `get_global_registry()` - Factory function, exported but never used
- `reset_global_registry()` - Utility function, exported but never used

**Status**: Exported from `src/processing/__init__.py` but:
- No imports found in any test files
- No imports found in any source files outside of services.py itself
- Only `ServiceRegistry` class is actually used (through its lazy properties)

**Recommendation**: Remove these unused functions and class, or keep them as planned future APIs

### 2. Unused Imports Found by Pylint

- `src/processing/materials/base.py:5` - Unused `ArrayLike` from numpy.typing
- `src/processing/core/abstracts.py:8` - Unused `List`, `Optional`, `Generic` from typing
- `src/processing/managers/cache.py:4` - Unused `logging` import
- `src/processing/managers/processor.py:5` - Unused `Dict` from typing
- `src/processing/managers/file.py:4` - Unused `logging` import
- `src/processing/resampling/service.py:13` - Unused `field` from dataclasses
- `src/processing/resampling/cache.py:21` - Unused `CACHE_DIR_DEFAULT` import

### 3. Unused Variables Found by Pylint

- `src/processing/interpolator.py:35` - Unused variable `nzp1`
- `src/processing/materials/properties.py:32` - Unused variable `converted`
- `src/processing/materials/properties.py:73` - Unused variable `converted`

## Test Results
All 1703 tests pass after removing the 3 dead code files.

## Recommendations
1. **High Priority**: Remove unused imports identified by Pylint (affects code clarity)
2. **Medium Priority**: Decide on `RockPhysicsService` and registry functions - either remove or document as future APIs
3. **Low Priority**: Fix unused variable assignments (rename to `_` if genuinely unused for side effects)
