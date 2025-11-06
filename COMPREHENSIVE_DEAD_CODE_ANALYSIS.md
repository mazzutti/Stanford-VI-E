# Comprehensive Dead Code Analysis - Entire src/ Folder

## Executive Summary
Found significant amounts of dead code across the codebase:
- **27+ unused imports** in analysis module alone
- **20+ unused variables** across various modules
- Multiple public APIs that are never imported/used

---

## 1. Unused Imports by Module

### Analysis Module (Most Issues)
- `src/analysis/mixins.py`: Dict, Protocol
- `src/analysis/results.py`: cast
- `src/analysis/config_builder.py`: asdict, Optional, Union
- `src/analysis/builder.py`: Callable
- `src/analysis/validator_chain.py`: Callable
- `src/analysis/processor_mixins.py`: ABC, abstractmethod, Generic, wraps
- `src/analysis/base.py`: Optional
- `src/analysis/types/base.py`: Enum, Callable
- `src/analysis/pipelines/factory.py`: Dict, ABC
- `src/analysis/pipelines/orchestrator.py`: timedelta
- `src/analysis/processors/config.py`: ClassVar
- `src/analysis/processors/registry.py`: Type, ABC, abstractmethod
- `src/analysis/processors/validators.py`: Tuple
- `src/analysis/processors/discrimination.py`: ProcessorConfig
- `src/analysis/processors/utils.py`: ProcessorConfig
- `src/analysis/facies/config.py`: field
- `src/analysis/facies/processor_setup.py`: Callable, Any
- `src/analysis/facies/analyzer.py`: Pipeline, create_facies_analysis_pipeline

### Other Modules
- `src/analysis/common.py`: os, sys, shutil
- (and more...)

---

## 2. Unused Variables by Module

### High Volume Areas
- `src/__main__.py` (5 unused variables):
  - Line 347: ni, nj, nz
  - Line 361: dt
  - Line 389: angle_gathers, full_stack_avo
  - Line 1056: DATA_PATH, FILE_MAP

- `src/plotting/slice_plotter.py` (6 unused variables):
  - Line 56: idx_j, idx_k
  - Line 93: idx_i, idx_k
  - Line 130: idx_i, idx_j

### Medium Volume Areas
- `src/analysis/pipelines/orchestrator.py`: stage_name
- `src/analysis/facies/stages.py`: analyzer
- `src/analysis/factories/builder.py`: proc_type
- `src/analysis/rock_physics/analyzer.py`: plotter
- `src/io/disk_cache.py`: k
- `src/modeling/processors.py`: nz
- `src/modeling/resampler.py`: dt
- `src/plotting/overlay_plotter.py`: nj, nk
- `src/signal/signal.py`: nk

---

## 3. Unused Public APIs (Never Imported)

### Important Finding - These are exported but never used:
- `src/__main__.py`: ParserFactory, main
- `src/analysis/builder.py`: Buildable
- `src/analysis/config_builder.py`: Configurable
- `src/analysis/common.py`: Multiple standard library re-exports
- `src/analysis/mixins.py`: ConfigurableMixin, StateTrackingMixin
- `src/analysis/domain/__init__.py`: Multiple domain handlers
- `src/analysis/cache/extractors.py`: ArrayExtractor, NpzExtractor, NpyExtractor
- Various model/result classes throughout

---

## 4. Recommendations by Priority

### High Priority (Code Quality)
1. Clean up unused imports across analysis/ module (27+ found)
   - Affects readability and code maintenance
   - Can be automated with pylint --fix-all or isort

2. Fix unused variable assignments (20+)
   - Rename to `_` or remove if genuinely unnecessary
   - Improves code clarity

### Medium Priority (API Design)
3. Audit unused public APIs
   - Decide: Keep as future APIs or remove?
   - Document intentional re-exports if needed

### Low Priority (Refactoring)
4. Review dead code patterns
   - Consider if Buildable, Configurable mixins should be used
   - Check if domain handlers are fully utilized

---

## 5. Scope of Work

### By Module
- **analysis/**: 20+ unused imports, 5+ unused variables
- **plotting/**: 8 unused variables
- **modeling/**: 2 unused variables
- **signal/**: 1 unused variable
- **io/**: 1 unused variable
- **src/__main__.py**: 5+ unused variables

### Total Estimated Issues
- **Unused Imports**: 30+
- **Unused Variables**: 20+
- **Unused Public APIs**: 15+

---

## Next Steps

1. **Automated cleanup**: Run pylint with --fix-all for imports
2. **Variable fixes**: Bulk fix with sed/replace_string_in_file
3. **API audit**: Review and document public APIs
4. **Re-test**: Ensure all 1703 tests pass after cleanup
