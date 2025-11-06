# Comprehensive Dead Code Analysis - Full src/ Folder

**Date**: November 6, 2025  
**Analysis Tool**: pylint with unused-import and unused-variable checks  
**Total Files Analyzed**: 278 Python files in src/

---

## Executive Summary

This analysis identified all dead code patterns in the entire src/ folder beyond what was already cleaned up in the processing module.

- **Total Unused Imports**: 21
- **Total Unused Variables**: 22
- **Total Issues**: 43

### Distribution by Module:
- **src/analysis/**: 16 unused imports + 4 unused variables = 20 issues
- **src/__main__.py**: 3 unused imports + 8 unused variables = 11 issues
- **src/plotting/**: 6 unused variables = 6 issues
- **src/modeling/**: 1 unused variable = 1 variable
- **src/io/**: 1 unused variable = 1 variable
- **src/signal/**: 1 unused variable = 1 variable

---

## 1. UNUSED IMPORTS (21 Total)

### 1.1 src/analysis/ Module (16 unused imports)

| File | Line | Import | Status |
|------|------|--------|--------|
| `processor_mixins.py` | 33 | `Generic` from typing | W0611 |
| `types/base.py` | 14 | `Enum` from enum | W0611 |
| `types/base.py` | 16 | `Callable` from typing | W0611 |
| `pipelines/factory.py` | 6 | `Dict` from typing | W0611 |
| `pipelines/factory.py` | 7 | `ABC` from abc | W0611 |
| `processors/registry.py` | 18 | `Type` from typing | W0611 |
| `processors/registry.py` | 27 | `ABC` from abc | W0611 |
| `processors/registry.py` | 27 | `abstractmethod` from abc | W0611 |
| `processors/validators.py` | 5 | `Tuple` from typing | W0611 |
| `facies/config.py` | 9 | `field` from dataclasses | W0611 |
| `facies/processor_setup.py` | 11 | `Callable` from typing | W0611 |
| `facies/processor_setup.py` | 11 | `Any` from typing | W0611 |
| `facies/stages.py` | 16 | `Optional` from typing | W0611 |
| `factories/validators.py` | 11 | `Optional` from typing | W0611 |
| `rock_physics/analyzer.py` | 334 | `PlotConfig` from src.plotting.helpers.config | W0611 |
| `validators.py` | 675 | `ValidationResult` from src.analysis.processors.config | W0611 |

### 1.2 src/__main__.py (3 unused imports)

| Line | Import | Status |
|------|--------|--------|
| 649 | `PlotlyPlotter` from src.plotting | W0611 |
| 679 | `SlicePlotter` from src.plotting | W0611 |
| 706 | `RockPhysicsPlotter` from src.plotting | W0611 |
| 876 | `PlotlyPlotter` from src.plotting (duplicate) | W0611 |

**Note**: PlotlyPlotter is imported twice (lines 649 and 876)

---

## 2. UNUSED VARIABLES (22 Total)

### 2.1 src/__main__.py (8 unused variables)

| Line | Variable | Context | Status |
|------|----------|---------|--------|
| 1056 | `DATA_PATH` | Multiple unpacking | W0612 |
| 1056 | `FILE_MAP` | Multiple unpacking | W0612 |
| 347 | `ni` | Loop unpacking `for ni, nj, nz in ...` | W0612 |
| 347 | `nj` | Loop unpacking | W0612 |
| 347 | `nz` | Loop unpacking | W0612 |
| 361 | `dt` | Unpacking from grid spec | W0612 |
| 389 | `angle_gathers` | Return value unpacking | W0612 |
| 389 | `full_stack_avo` | Return value unpacking | W0612 |

### 2.2 src/plotting/ (6 unused variables)

| File | Line | Variable | Context | Status |
|------|------|----------|---------|--------|
| `overlay_plotter.py` | 75 | `nj` | Loop unpacking `for nj, nk in ...` | W0612 |
| `overlay_plotter.py` | 75 | `nk` | Loop unpacking | W0612 |
| `slice_plotter.py` | 56 | `idx_j` | Loop unpacking `for idx_i, idx_j, idx_k in ...` | W0612 |
| `slice_plotter.py` | 56 | `idx_k` | Loop unpacking | W0612 |
| `slice_plotter.py` | 93 | `idx_i` | Loop unpacking `for idx_i, idx_k in ...` | W0612 |
| `slice_plotter.py` | 93 | `idx_k` | Loop unpacking | W0612 |
| `slice_plotter.py` | 130 | `idx_i` | Loop unpacking `for idx_i, idx_j in ...` | W0612 |
| `slice_plotter.py` | 130 | `idx_j` | Loop unpacking | W0612 |

### 2.3 src/analysis/ (4 unused variables)

| File | Line | Variable | Context | Status |
|------|------|----------|---------|--------|
| `facies/stages.py` | 244 | `analyzer` | Variable assignment | W0612 |
| `factories/builder.py` | 980 | `proc_type` | Loop variable | W0612 |
| `pipelines/orchestrator.py` | 391 | `stage_name` | Loop variable | W0612 |
| `rock_physics/analyzer.py` | 503 | `plotter` | Variable assignment | W0612 |

### 2.4 src/modeling/ (1 unused variable)

| File | Line | Variable | Context | Status |
|------|------|----------|---------|--------|
| `processors.py` | 110 | `nz` | Loop unpacking `for nz in ...` | W0612 |
| `resampler.py` | 53 | `dt` | Variable assignment | W0612 |

### 2.5 src/io/ (1 unused variable)

| File | Line | Variable | Context | Status |
|------|------|----------|---------|--------|
| `disk_cache.py` | 246 | `k` | Loop variable | W0612 |

### 2.6 src/signal/ (1 unused variable)

| File | Line | Variable | Context | Status |
|------|------|----------|---------|--------|
| `signal.py` | 94 | `nk` | Loop unpacking | W0612 |

---

## 3. CLEANUP RECOMMENDATIONS BY PRIORITY

### High Priority (Code Quality Issues)

#### 3.1 Fix Unused Loop Variables in Unpacking

These should be replaced with `_` underscore to indicate intentionally unused values:

```python
# src/plotting/slice_plotter.py:56
# BEFORE:
for idx_i, idx_j, idx_k in some_iterator:
    # idx_j and idx_k not used

# AFTER:
for idx_i, _, _ in some_iterator:
```

**Affected Files**:
- `src/plotting/slice_plotter.py` (5 instances at lines 56, 93, 130)
- `src/plotting/overlay_plotter.py` (2 instances at line 75)
- `src/__main__.py` (5 instances at lines 347, 1056)
- `src/signal/signal.py` (1 instance at line 94)
- `src/modeling/processors.py` (1 instance at line 110)
- `src/modeling/resampler.py` (1 instance at line 53)
- `src/io/disk_cache.py` (1 instance at line 246)

**Total**: 17 unused loop variables to fix

#### 3.2 Remove Unused Variable Assignments

These are standalone assignments that are never referenced:

**Files to Review**:
- `src/analysis/facies/stages.py:244` - `analyzer` assignment
- `src/analysis/factories/builder.py:980` - `proc_type` loop variable
- `src/analysis/pipelines/orchestrator.py:391` - `stage_name` loop variable
- `src/analysis/rock_physics/analyzer.py:503` - `plotter` assignment

**Action**: Remove if truly unused, or verify if these are intentional placeholders.

### Medium Priority (Import Cleanup)

#### 3.3 Remove Unused Imports from src/analysis/

**Most Issues** (files with multiple unused imports):
- `src/analysis/processor_mixins.py`: Remove `Generic` (line 33)
- `src/analysis/types/base.py`: Remove `Enum` (line 14), `Callable` (line 16)
- `src/analysis/pipelines/factory.py`: Remove `Dict` (line 6), `ABC` (line 7)
- `src/analysis/processors/registry.py`: Remove `Type`, `ABC`, `abstractmethod` (lines 18, 27)

**Single Unused Imports**:
- `src/analysis/processors/validators.py`: Remove `Tuple` (line 5)
- `src/analysis/facies/config.py`: Remove `field` (line 9)
- `src/analysis/facies/processor_setup.py`: Remove `Callable`, `Any` (line 11)
- `src/analysis/facies/stages.py`: Remove `Optional` (line 16)
- `src/analysis/factories/validators.py`: Remove `Optional` (line 11)
- `src/analysis/rock_physics/analyzer.py`: Remove `PlotConfig` (line 334)
- `src/analysis/validators.py`: Remove `ValidationResult` (line 675)

**Total**: 16 unused imports in src/analysis/

#### 3.4 Fix Duplicate and Unused Imports in src/__main__.py

- **Lines 649 & 876**: Duplicate import of `PlotlyPlotter`
  - Keep one, remove the other
- **Line 679**: `SlicePlotter` imported but not used
- **Line 706**: `RockPhysicsPlotter` imported but not used

**Recommendation**: Remove lines 679, 706, and 876 (keep line 649 if actually used)

---

## 4. SCOPE OF WORK

### Estimated Effort (Manual Approach - Recommended)

1. **Loop Variable Fixes** (~15 minutes)
   - Replace unused loop variables with `_`
   - 17 instances across 8 files
   - Low risk, easy to verify

2. **Unused Variable Assignment Removal** (~20 minutes)
   - Review 4 assignments for context
   - Determine if truly unused or intentional placeholders
   - Remove if unused

3. **src/analysis/ Import Cleanup** (~30 minutes)
   - Remove 16 unused imports
   - Test after each file or in batches
   - Higher risk due to complexity of analysis module

4. **src/__main__.py Import Cleanup** (~10 minutes)
   - Remove duplicate PlotlyPlotter import
   - Remove unused SlicePlotter and RockPhysicsPlotter imports
   - Low risk

**Total Estimated Time**: ~75 minutes (1 hour 15 minutes)

---

## 5. IMPLEMENTATION STRATEGY

### Phase 1: Quick Wins (30 minutes)
1. Fix all unused loop variables with `_` replacements
2. Verify tests still pass

### Phase 2: Remove Unused Imports (25 minutes)
1. Start with `src/__main__.py` (low complexity)
2. Move to `src/plotting/` and `src/modeling/` 
3. Finally tackle `src/analysis/` (highest complexity)
4. Test after each batch of 2-3 files

### Phase 3: Verify (10 minutes)
1. Run full test suite: `pytest -xvs`
2. Run pylint verification: `pylint --disable=all --enable=unused-import,unused-variable src/`
3. Ensure no regressions

---

## 6. RISK ASSESSMENT

| Category | Risk | Mitigation |
|----------|------|-----------|
| Loop variable fixes | Very Low | Replace with `_` only, no logic changes |
| Unused variable removal | Low | Review context first, remove only dead code |
| Import removal in analysis/ | Medium | Complex module, test thoroughly |
| Overall | Low | Incremental approach with testing |

---

## 7. TESTING CHECKLIST

After each cleanup phase:

- [ ] `pytest -xvs` - All tests pass
- [ ] `pylint --disable=all --enable=unused-import,unused-variable src/` - No new warnings
- [ ] `git diff` - Review all changes
- [ ] Manual smoke test of affected modules

---

## Files Requiring Changes (Priority Order)

### Tier 1: Quick Wins (Low Risk)
1. `src/plotting/slice_plotter.py` - Fix 5 unused loop variables
2. `src/plotting/overlay_plotter.py` - Fix 2 unused loop variables
3. `src/signal/signal.py` - Fix 1 unused loop variable
4. `src/modeling/resampler.py` - Fix 1 unused loop variable
5. `src/modeling/processors.py` - Fix 1 unused loop variable

### Tier 2: Import Cleanup (Medium Risk)
1. `src/__main__.py` - Remove 3 unused imports + fix 8 unused variables
2. `src/io/disk_cache.py` - Fix 1 unused loop variable

### Tier 3: Analysis Module (Higher Risk)
1. `src/analysis/processor_mixins.py` - Remove 1 unused import
2. `src/analysis/types/base.py` - Remove 2 unused imports
3. `src/analysis/pipelines/factory.py` - Remove 2 unused imports
4. `src/analysis/processors/registry.py` - Remove 3 unused imports
5. `src/analysis/processors/validators.py` - Remove 1 unused import
6. `src/analysis/facies/config.py` - Remove 1 unused import + fix 1 unused variable
7. `src/analysis/facies/processor_setup.py` - Remove 2 unused imports
8. `src/analysis/facies/stages.py` - Remove 1 unused import + fix 1 unused variable
9. `src/analysis/factories/validators.py` - Remove 1 unused import
10. `src/analysis/factories/builder.py` - Fix 1 unused variable
11. `src/analysis/pipelines/orchestrator.py` - Fix 1 unused variable
12. `src/analysis/rock_physics/analyzer.py` - Remove 1 unused import + fix 1 unused variable
13. `src/analysis/validators.py` - Remove 1 unused import

---

## Previous Session Context

This is a continuation of the dead code cleanup that successfully cleaned up `src/processing/` module:
- ✅ Deleted 3 unused files
- ✅ Removed 7 unused imports from 7 files
- ✅ Fixed 3 unused variables
- ✅ All 1703 tests passing

**Lessons Learned**: Manual, surgical approach file-by-file is safer than automated bulk cleanup.

---

## Next Steps

1. Implement Phase 1 fixes (loop variables)
2. Run full test suite after each file
3. Create commit for each completed file
4. Continue with Phase 2 (imports)
5. Final verification and commit

---

**Generated**: November 6, 2025  
**Session Type**: Comprehensive Dead Code Analysis  
**Analysis Method**: pylint with W0611 (unused-import) and W0612 (unused-variable) checks
