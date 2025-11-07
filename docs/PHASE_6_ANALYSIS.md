# Phase 6: Processor Consolidation - ANALYSIS

## Overview
Phase 6 will consolidate repeated processor patterns and eliminate duplication in processor implementations across the codebase. Target: -1,500-2,000 LOC.

## Key Findings

### 1. DUPLICATE PROCESSOR ABSTRACT BASE CLASSES
**Critical Finding**: Two identical Processor ABC implementations
- **File 1**: `src/analysis/processors/base.py` (159 LOC)
  - Class: `Processor(ABC)` with `process(*args, **kwargs)` abstract method
  - Also contains: `BaseProcessor(Processor)` with smart delegation
  
- **File 2**: `src/processing/core/abstracts.py` (187 LOC)
  - Class: `Processor(ABC)` with `process(data, **kwargs)` abstract method
  - Simpler implementation, no delegation logic

**Usage Pattern**:
- `src/analysis/processors/base.py` -> Used by facies, rock_physics, analysis modules
- `src/processing/core/abstracts.py` -> Used by material properties, resampling, validators

**Consolidation Strategy**: 
- Keep analysis.processors.base.Processor as primary (more sophisticated with delegation)
- Update processing.core.abstracts to re-export or inherit from analysis version
- Estimated savings: -50 LOC (remove duplicate class, update imports)

### 2. PROCESSOR MIXIN DUPLICATION
**File**: `src/analysis/processor_mixins.py` (751 LOC)

Contains mixins:
- LoggingMixin: Logging behavior
- CachingMixin: Caching functionality  
- ValidationMixin: Input/output validation
- StateTrackingMixin: State management
- ErrorHandlingMixin: Error handling
- MetricsMixin: Metrics collection
- ProcessorMixinManager: Manager class

**Issue**: These are general-purpose mixins that could be moved to src/core or made more reusable

**Consolidation Opportunity**:
- Could be refactored to use composition instead of mixins
- Could share with other components that need logging/caching/validation
- Potential savings: -100-200 LOC through consolidation

### 3. MODELING PROCESSORS
**File**: `src/modeling/processors.py` (123 LOC)
- ReflectivityComputer: Computes reflectivity using Zoeppritz
- WaveletConvolver: 3D wavelet convolution

**Issue**: These are specialized, not duplicated. No consolidation needed.

### 4. SERVICE FACTORY PATTERNS
**File**: `src/analysis/factories/service_factory.py` (640 LOC)
- ProcessorServiceFactory: Creates processor instances

**Related**: 
- ProcessorRegistry: Manages processor metadata and registration (416 LOC)
- ProcessorDecorators: Decorator utilities (195 LOC)
- ProcessorConfig: Configuration management (247 LOC)

**Consolidation Opportunity**:
- These files work together to manage processor lifecycle
- Could be consolidated into unified processor management module
- Potential savings: -200-300 LOC through consolidation

### 5. ANALYSIS-SPECIFIC PROCESSORS
**Location**: `src/analysis/processors/`
- boundary.py: Boundary detection (334 LOC)
- utils.py: Utilities (433 LOC)  
- config.py: Configuration (247 LOC)
- registry.py: Registry (416 LOC)
- decorators.py: Decorators (195 LOC)
- base.py: Base classes (159 LOC)

**Total in processors/**: ~1,784 LOC

**Consolidation Opportunity**:
- Could consolidate utils, config, registry into single management module
- Estimated savings: -400-500 LOC through better organization

## Implementation Plan for Phase 6

### Phase 6a: Unify Processor ABCs (Priority 1)
1. Update `src/processing/core/abstracts.py` to use `src/analysis/processors.base.Processor`
2. Remove duplicate ABC definition
3. Update all imports
4. **Estimated savings**: -50 LOC

### Phase 6b: Consolidate Processor Registry/Config/Utils (Priority 2)
1. Merge registry, config, and utils into unified processor management module
2. Reduce redundancy in processor management code
3. **Estimated savings**: -300-400 LOC

### Phase 6c: Optimize Processor Mixins (Priority 3)
1. Extract reusable mixin patterns
2. Move to core if applicable
3. Consolidate redundant functionality
4. **Estimated savings**: -100-200 LOC

### Phase 6d: Service Factory Consolidation (Priority 4)
1. Simplify ProcessorServiceFactory
2. Reduce boilerplate in factory methods
3. **Estimated savings**: -200-300 LOC

## Total Phase 6 Target: -1,500-2,000 LOC

Breakdown:
- 6a (Unify ABCs): -50 LOC
- 6b (Registry/Config/Utils): -400 LOC
- 6c (Mixins): -150 LOC
- 6d (Factory): -250 LOC
- Additional optimization: -650-1,150 LOC

## Risks and Mitigations

**Risk 1**: Circular imports between analysis and processing modules
**Mitigation**: Use TYPE_CHECKING for forward references, lazy imports

**Risk 2**: Breaking existing processor implementations
**Mitigation**: Extensive testing, maintain backward compatibility

**Risk 3**: Multiple processor base classes in different modules
**Mitigation**: Unify early, create clear import paths

## Next Steps
1. Implement Phase 6a first (easiest, low-risk)
2. Test thoroughly before proceeding
3. Then tackle 6b, 6c, 6d in order
