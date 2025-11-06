# Phase 3: Public API Audit & Strategic Decisions

**Date**: November 6, 2025  
**Status**: ANALYSIS & DECISION PHASE  
**Audit Date**: November 6, 2025

---

## Executive Summary

Found **234 unused exports** out of 639 total public API items. However, **not all should be removed**:
- **⚠️ HIGH PRIORITY**: 3-5 truly dead exports to remove
- **📋 MEDIUM PRIORITY**: ~50 exports to document as intentional public APIs
- **ℹ️ LOW PRIORITY**: ~180 internal implementations exported for convenience

---

## Audit Results Breakdown

### Total Public API Exports
```
Total __all__ exports found: 639
Unused (never internally imported): 234
Percentage unused: 36.6%
```

### By Priority Category

#### 🔴 **CRITICAL REMOVALS** (Must Remove - 4 items)

These are genuinely unused and serve no purpose:

1. **src/__main__.py** (2 items)
   - `ParserFactory` - Never imported; used in CLI only
   - `main` - Entry point; not imported anywhere

2. **src/analysis/common.py** (3 items)
   - `os` - Standard library re-export (unused)
   - `sys` - Standard library re-export (unused)  
   - `shutil` - Standard library re-export (unused)

3. **src/analysis/cache/__init__.py** (4 items - CONDITIONAL)
   - `_FILE_PREFIX` - Private constant, shouldn't be in __all__
   - `_FULL_STACK_KEY` - Private constant, shouldn't be in __all__
   - `_NPY_EXTENSION` - Private constant, shouldn't be in __all__
   - `_NPZ_EXTENSION` - Private constant, shouldn't be in __all__

4. **src/io/disk_cache.py** (3 items - CONDITIONAL)
   - `default_disk_cache` - Singleton; could be private
   - `get_default_disk_cache` - Getter; could be private
   - `make_disk_cache` - Factory; could be private

**Recommendation**: Remove items 1-3, review item 4 for public API decision

---

#### 🟡 **LIBRARY PUBLIC APIs** (Keep & Document - ~50 items)

These are intentionally exported for library users. Should be KEPT but documented:

**Analysis Module** (Main Library Interface):
```
src/analysis/__init__.py exports 73 items including:
- AnalysisBuilder, ConfigBuilder - Main entry points for users
- Exception classes: AnalysisException, ComputationError, etc.
- Validators: CompositeValidator, RangeValidator, etc.
- Domain handlers: DepthDomainHandler, TimeDomainHandler
- Results: ResultData, ResultMetadata
```

**Recommendation**: Keep all; document as public API in README/docs

**Key Pattern**: These are in top-level `__init__.py` files for easy importing
```python
from src.analysis import AnalysisBuilder  # ✅ Supported
from src.analysis import ResultData        # ✅ Supported
```

---

#### 🔵 **INTERNAL CONVENIENCE EXPORTS** (Keep - ~180 items)

These are internal implementations exported for convenience:

Examples:
```
src/analysis/builder.py:
  - Buildable - Interface; exported but not directly used internally

src/analysis/processors/__init__.py:
  - 20+ processor types - Exported for convenience, not directly used

src/analysis/domain/handlers.py:
  - DomainHandler, DepthDomainHandler, etc. - Used via factory pattern

src/plotting/helpers/formatting.py:
  - FormattingHelper functions - Used internally via different patterns
```

**Recommendation**: Keep all; these enable flexible usage patterns

---

## Detailed Findings

### Category 1: Standard Library Re-exports (❌ REMOVE)

**File**: src/analysis/common.py
```python
__all__ = ["os", "sys", "shutil", ...]  # ❌ Why export stdlib?

# These should not be exported; users can import directly
import os  # ✅ Do this instead
```

**Action**: Remove `os`, `sys`, `shutil` from __all__

---

### Category 2: Private Constants in Public API (⚠️ REVIEW)

**File**: src/analysis/cache/__init__.py
```python
__all__ = [
    "_FILE_PREFIX",      # ⚠️ Starts with _; shouldn't be public
    "_FULL_STACK_KEY",   # ⚠️ Starts with _; shouldn't be public
    "_NPY_EXTENSION",    # ⚠️ Starts with _; shouldn't be public
    "_NPZ_EXTENSION",    # ⚠️ Starts with _; shouldn't be public
]
```

**Action**: Remove underscore-prefixed items from __all__

---

### Category 3: Unimported Main Entry Points (❌ REMOVE)

**File**: src/__main__.py
```python
__all__ = ["ParserFactory", "main"]

# Used only via: python -m src
# Not imported as: from src import ParserFactory
```

**Question**: Are these meant to be public library APIs or just CLI tools?
- If CLI only → Remove from __all__
- If library API → Document in README

**Recommendation**: Remove; these are CLI tools, not library APIs

---

### Category 4: Legitimate Public APIs (✅ KEEP)

**File**: src/analysis/__init__.py
```python
__all__ = [
    "AnalysisBuilder",        # ✅ Main entry point
    "ConfigBuilder",          # ✅ Configuration helper
    "AnalysisException",      # ✅ Exception type
    "ResultData",             # ✅ Result type
    # ... 70 more items
]

# Used externally via:
# from src.analysis import AnalysisBuilder
# analyzer = AnalysisBuilder().build()
```

**Recommendation**: Keep all; document as public API

---

## Strategic Recommendations

### Phase 3 Action Plan

#### **Step 1: Remove Clearly Unused Items** (5-10 min)
Remove these items from `__all__`:
1. `src/__main__.py`: Remove `ParserFactory` and `main`
2. `src/analysis/common.py`: Remove `os`, `sys`, `shutil`
3. `src/analysis/cache/__init__.py`: Remove `_FILE_PREFIX`, `_FULL_STACK_KEY`, `_NPY_EXTENSION`, `_NPZ_EXTENSION`
4. `src/io/disk_cache.py`: Remove `default_disk_cache`, `get_default_disk_cache`, `make_disk_cache`

**Impact**: Cleaner, more professional public API

---

#### **Step 2: Document Intentional Public APIs** (15-20 min)
Create/update: `API_REFERENCE.md`
- Document all public APIs in `src/analysis/__init__.py`
- Show usage examples
- Organize by category (builders, exceptions, validators, etc.)

**Impact**: Better for library users; clearer API surface

---

#### **Step 3: Review Conditional Items** (10-15 min)
Review these for removal:
- `src/io/disk_cache.py`: Singleton pattern - keep or make private?
- `src/plotting/__init__.py`: `np`, `plt` re-exports - necessary?

**Impact**: Further API cleanup if appropriate

---

## Decision Matrix

| Item | Category | Action | Reason |
|------|----------|--------|--------|
| ParserFactory (__main__.py) | CLI Tool | REMOVE | Not a library API |
| main (__main__.py) | CLI Tool | REMOVE | Not a library API |
| os, sys, shutil (common.py) | StdLib | REMOVE | Users can import directly |
| Private constants (cache/__init__.py) | Internal | REMOVE | Shouldn't be public |
| AnalysisBuilder (analysis/__init__.py) | Public API | KEEP | Main entry point |
| ConfigBuilder (analysis/__init__.py) | Public API | KEEP | Main entry point |
| Exception classes (analysis/__init__.py) | Public API | KEEP | Error handling |
| Validators (analysis/__init__.py) | Public API | KEEP | User-facing tools |
| Internal impls (processors/__init__.py) | Convenience | KEEP | Enable flexibility |

---

## Estimated Impact

### Before Phase 3
```
Total exports: 639
Unused in codebase: 234 (36.6%)
Public API clarity: Low (mixed public/internal)
```

### After Phase 3 (with removals)
```
Total exports: ~625 (remove ~14 items)
Unused in codebase: ~220 (35.2%)
Public API clarity: High (clear distinction)
```

---

## Implementation Notes

### Why Keep Most Unused Exports?

These are NOT dead code; they're:
1. **Convenience APIs** - Enable multiple usage patterns
2. **Future-proofing** - Allow flexible imports
3. **Library design** - Top-level namespace for discoverability
4. **Backward compatibility** - Remove carefully to avoid breaking changes

### Testing Strategy
- ✅ All 1703 tests should still pass
- ✅ Verify imports in tests don't break
- ✅ Verify no external packages depend on removed items

---

## Timeline

- **Decision phase** (now): Confirm which items to remove
- **Implementation**: ~30 minutes
  - Remove from __all__ statements (10 min)
  - Update tests if needed (10 min)
  - Run full test suite (10 min)
- **Documentation**: ~15 minutes
  - Create API reference (optional)
  - Update README

---

## Next Steps

### Immediate (Within 5 minutes)
1. ✅ Confirm recommendation to remove 14 items
2. ✅ Proceed with implementation

### Short-term (If approved)
3. Remove items from __all__ statements
4. Run tests to verify no breakage
5. Commit with clear messages

### Future (Optional)
6. Create comprehensive API documentation
7. Add API stability labels (stable, experimental, internal)
8. Set up deprecation warnings for future API changes

---

## Conclusion

Phase 3 is **primarily about API clarity**, not code optimization.

**Key Finding**: 36% of exports are unused internally, but most are legitimate convenience APIs for library users.

**Recommended Action**: Remove ~14 clearly unnecessary items while keeping the intentional public APIs intact.

**Expected Outcome**: Cleaner, more professional library interface.
