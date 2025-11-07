"""
# Pattern Integration Guide: Dependency Injection, Event Bus, Circuit Breaker, Retry

This document provides a comprehensive guide to the integrated design patterns
in the Stanford-VI-E analysis framework.

## Overview

Four critical design patterns have been integrated into the production code:

1. **Dependency Injection (DI)**: Loose coupling through service container
2. **Event Bus**: Decoupled event handling with publish-subscribe pattern
3. **Circuit Breaker**: Fault tolerance with automatic state management
4. **Retry**: Automatic resilience with exponential backoff

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AnalysisSystem                          │
│  (Main entry point for pattern-integrated analysis)        │
└────────┬────────────────────────────────────────────────────┘
         │
         ├─► ServiceContainer (Dependency Injection)
         │   ├─ EventBus
         │   ├─ ConfigManager
         │   ├─ CircuitBreakerPool
         │   ├─ FaciesCorrelationAnalyzer
         │   └─ Other services
         │
         ├─► IntegratedAnalyzer
         │   ├─ Event Bus Integration
         │   ├─ Circuit Breaker Management
         │   ├─ DI Service Resolution
         │   └─ Retry Support
         │
         └─► ComponentFactory
             └─ Creates components with all patterns
```

## Pattern Details

### 1. Dependency Injection

**Location**: `src/analysis/service_container.py`

**Purpose**: Decouple services from their implementations

**Components**:
- `Container`: DI container for service registration
- `ServiceProvider`: Resolves registered services
- `Lifecycle`: Manages service lifecycles (singleton, transient, scoped)

**Usage**:

```python
from src.analysis.service_container import create_container

# Create container
container = create_container()
provider = container.create_service_provider()

# Resolve services
analyzer = provider.resolve("FaciesCorrelationAnalyzer")
config_manager = provider.resolve("ConfigManager")
event_bus = provider.resolve("EventBus")
```

**Benefits**:
- Easy testing with mock services
- Service configuration centralization
- Reduced coupling between components
- Easy service lifetime management

### 2. Event Bus

**Location**: `src/analysis/patterns/event_bus.py`

**Purpose**: Decouple event publishers from subscribers

**Components**:
- `EventBus`: Central event dispatcher
- `Event`: Base class for all events
- `EventHandler`: Handler registration and execution
- `EventFilter`: Fine-grained event filtering

**Application Events**: `src/analysis/events.py`
- `AnalysisStartedEvent`
- `AnalysisCompletedEvent`
- `AnalysisFailedEvent`
- `CacheHitEvent`
- `ProcessorExecutionStartedEvent`
- And more...

**Usage**:

```python
from src.analysis.integration import AnalysisSystem

system = AnalysisSystem()

# Subscribe to events (fluent API)
system.subscribe_to_events() \
    .on_analysis_started(lambda e: print(f"Started: {e.analysis_type}")) \
    .on_analysis_completed(lambda e: print(f"Completed: {e.result_summary}")) \
    .on_error(lambda e: print(f"Error: {e.error_message}"))

# Publish events manually
from src.analysis import events
event = events.AnalysisStartedEvent(
    analysis_type="FaciesCorrelation",
    domain="depth",
    cache_dir=".cache",
)
system.get_event_bus().publish(event)
```

**Benefits**:
- Decoupled event handling
- Multiple independent subscribers
- Event filtering and routing
- Asynchronous event processing ready

### 3. Circuit Breaker

**Location**: `src/analysis/patterns/circuit_breaker.py`

**Purpose**: Fault tolerance through failure detection and recovery

**States**:
- **CLOSED**: Normal operation (requests pass through)
- **OPEN**: Failure threshold reached (requests rejected)
- **HALF_OPEN**: Recovery attempt (limited requests allowed)

**Components**:
- `CircuitBreaker`: Main circuit breaker state machine
- `CircuitBreakerPool`: Manages multiple circuit breakers
- `@circuit_breaker`: Decorator for function protection

**Usage**:

```python
from src.analysis.patterns.circuit_breaker import circuit_breaker

# Decorator usage
@circuit_breaker(
    name="analysis_operation",
    failure_threshold=5,
    recovery_timeout=60,
)
def risky_operation():
    # This will be protected from cascading failures
    pass

# Or programmatic usage
from src.analysis.patterns.circuit_breaker import CircuitBreaker

breaker = CircuitBreaker(
    name="api_call",
    failure_threshold=3,
    recovery_timeout=30,
)

try:
    result = breaker.call(unreliable_function)
except CircuitBreakerOpen:
    print("Circuit is open, retrying later...")
```

**Integrated with Facies Analyzer**:

```python
# In src/analysis/facies/analyzer.py
@circuit_breaker(
    name="facies_correlation_analysis",
    failure_threshold=3,
    recovery_timeout=60,
)
def run(self, *, cache_dir: str, domain: Domain, verbose: bool = False):
    # Analysis is protected by circuit breaker
    pass
```

**Benefits**:
- Prevents cascading failures
- Automatic recovery attempts
- State tracking and monitoring
- Configurable failure thresholds

### 4. Retry

**Location**: `src/analysis/patterns/retry.py`

**Purpose**: Automatic resilience with configurable backoff strategies

**Strategies**:
- **ExponentialBackoffStrategy**: 2^attempt * initial_delay
- **LinearBackoffStrategy**: attempt * initial_delay
- **FibonacciBackoffStrategy**: Fibonacci sequence delays
- **ConstantBackoffStrategy**: Fixed delays

**Components**:
- `RetryPolicy`: Core retry implementation
- `@retry`: Decorator for function protection
- `RetryStrategy`: Abstract base for delay calculation

**Usage**:

```python
from src.analysis.patterns.retry import retry, ExponentialBackoffStrategy

# Decorator usage
@retry(
    max_attempts=3,
    initial_delay=1.0,
    strategy=ExponentialBackoffStrategy(),
    retryable_exceptions=[RuntimeError, IOError],
)
def unreliable_operation():
    # This will be retried up to 3 times with exponential backoff
    pass

# Or programmatic usage
from src.analysis.patterns.retry import RetryPolicy

policy = RetryPolicy(
    max_attempts=3,
    initial_delay=1.0,
    strategy=ExponentialBackoffStrategy(),
    name="api_call",
)

result = policy.execute(unreliable_function, arg1, arg2)
```

**Integrated with Facies Analyzer**:

```python
# In src/analysis/facies/analyzer.py
@retry(
    max_attempts=3,
    initial_delay=2.0,
    retryable_exceptions=[RuntimeError, OSError, IOError],
)
def run(self, *, cache_dir: str, domain: Domain, verbose: bool = False):
    # Analysis will be retried automatically on failure
    pass
```

**Benefits**:
- Handles transient failures automatically
- Configurable backoff strategies
- Prevents thundering herd with jitter
- Reduces manual error handling code

## Integration Points

### IntegratedAnalyzer

The `IntegratedAnalyzer` class serves as the main integration point:

```python
from src.analysis.integrated_analyzer import IntegratedAnalyzer
from src.analysis.patterns.event_bus import EventBus
from src.analysis.service_container import create_container

# Create with all patterns
event_bus = EventBus()
container = create_container()

analyzer = IntegratedAnalyzer(
    event_bus=event_bus,
    container=container,
    service_provider=container.create_service_provider(),
)

# Use DI to resolve services
config_manager = analyzer.get_service("ConfigManager")

# Publish events
from src.analysis import events
event = events.AnalysisStartedEvent(...)
analyzer.publish_event(event)

# Use circuit breakers
breaker = analyzer.get_circuit_breaker("my_operation")
```

### AnalysisSystem

The `AnalysisSystem` class provides a high-level API:

```python
from src.analysis.integration import AnalysisSystem, SystemConfiguration

# Create system
config = SystemConfiguration()
config.max_retries = 5
config.circuit_breaker_threshold = 3

system = AnalysisSystem(configuration=config)

# Create analyzer with patterns
analyzer = system.create_analyzer()

# Subscribe to events
system.subscribe_to_events() \
    .on_analysis_started(my_handler) \
    .on_error(error_handler)

# Get services
event_bus = system.get_service("EventBus")

# Reset circuit breakers
system.reset_circuit_breakers()
```

### ComponentFactory

The `ComponentFactory` creates components with patterns:

```python
from src.analysis.factory import ComponentFactory

factory = ComponentFactory()

# Create analyzer with all patterns
analyzer = factory.create_analyzer()

# Create resilient processor
def my_processor():
    pass

resilient_processor = factory.create_resilient_processor(
    my_processor,
    max_retries=3,
    circuit_breaker_threshold=5,
)
```

## Usage Examples

### Example 1: Basic Analysis with Event Handling

```python
from src.analysis.integration import AnalysisSystem

# Create system
system = AnalysisSystem()

# Subscribe to events
def on_complete(event):
    print(f"Analysis complete: {event.result_summary}")

system.subscribe_to_events().on_analysis_completed(on_complete)

# Create analyzer
analyzer = system.create_analyzer()

# Run analysis (with circuit breaker and retry protection)
result = analyzer.run_with_command(cache_dir=".cache", domain="depth")
```

### Example 2: Service Resolution with DI

```python
from src.analysis.service_container import ServiceContainerBuilder

# Build container
builder = ServiceContainerBuilder()
container = builder \
    .with_event_bus() \
    .with_circuit_breaker_pool() \
    .with_config_manager() \
    .with_facies_analyzer() \
    .build()

# Resolve services
provider = container.create_service_provider()
analyzer = provider.resolve("FaciesCorrelationAnalyzer")
config = provider.resolve("ConfigManager")
```

### Example 3: Custom Resilient Operation

```python
from src.analysis.factory import create_resilient_processor

# Create resilient operation
def analyze_data(data):
    # Do something with data
    return result

resilient_analyze = create_resilient_processor(
    analyze_data,
    max_retries=5,
    retry_delay=2.0,
    circuit_breaker_threshold=3,
)

# Use with automatic resilience
result = resilient_analyze(data)
```

### Example 4: Manual Circuit Breaker Management

```python
from src.analysis.patterns.circuit_breaker import get_circuit_breaker

# Get breaker
breaker = get_circuit_breaker("my_operation")

# Check state
if breaker.state.value == "OPEN":
    print("Circuit is open, backing off...")
else:
    # Get statistics
    stats = breaker.get_stats()
    print(f"Success rate: {stats.success_rate}%")

# Reset manually
breaker.reset()
```

## Configuration

### SystemConfiguration

Control pattern behavior through `SystemConfiguration`:

```python
from src.analysis.integration import SystemConfiguration

config = SystemConfiguration()

# Retry settings
config.max_retries = 5
config.retry_delay = 2.0

# Circuit breaker settings
config.circuit_breaker_threshold = 3
config.circuit_breaker_timeout = 120

# Pattern enablement
config.enable_event_bus = True
config.enable_circuit_breaker = True
config.enable_retry = True
config.enable_dependency_injection = True
```

## Testing

### Mocking with DI

```python
from src.analysis.service_container import ServiceContainerBuilder
from unittest.mock import Mock

# Create container with mock services
mock_event_bus = Mock()
mock_config = Mock()

builder = ServiceContainerBuilder()
builder.with_event_bus(mock_event_bus)
builder.with_config_manager(mock_config)

container = builder.build()

# Use container in tests
analyzer = IntegratedAnalyzer(container=container)
```

### Event Testing

```python
from src.analysis.patterns.event_bus import EventBus
from src.analysis import events

# Create bus
bus = EventBus()

# Capture events
received_events = []

def capture_event(event):
    received_events.append(event)

bus.subscribe(events.AnalysisEventType.STARTED.value, capture_event)

# Test event publishing
event = events.AnalysisStartedEvent(...)
bus.publish(event)

assert len(received_events) == 1
```

### Circuit Breaker Testing

```python
from src.analysis.patterns.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
import pytest

breaker = CircuitBreaker(failure_threshold=2)

# Simulate failures
with pytest.raises(RuntimeError):
    breaker.call(lambda: 1/0)

with pytest.raises(RuntimeError):
    breaker.call(lambda: 1/0)

# Circuit should be open
with pytest.raises(CircuitBreakerOpen):
    breaker.call(lambda: "success")
```

## Performance Considerations

1. **Event Bus**: Synchronous by default; asynchronous version available
2. **Circuit Breaker**: Minimal overhead; state checked on each call
3. **Retry**: Exponential backoff prevents resource exhaustion
4. **DI Container**: Caching and singleton support for efficiency

## Migration Guide

### For Existing Code

1. **Add Event Bus**:
   ```python
   analyzer.set_event_bus(EventBus())
   ```

2. **Use DI Container**:
   ```python
   container = create_container()
   provider = container.create_service_provider()
   service = provider.resolve("ServiceName")
   ```

3. **Add Resilience**:
   ```python
   @circuit_breaker(name="operation")
   @retry(max_attempts=3)
   def operation():
       pass
   ```

## Troubleshooting

### Circuit Breaker Always Open

Check:
- Failure threshold setting
- Exception types being caught
- Recovery timeout value

### Events Not Received

Check:
- Subscription to correct event type
- Event bus instance same as published events
- Handler function signature correct

### Service Resolution Failed

Check:
- Service registered in container
- Dependency names correct
- Lifecycle configuration correct

## Files and Locations

```
src/analysis/
├── service_container.py          # DI container setup
├── factory.py                    # Component factory
├── integration.py                # AnalysisSystem integration
├── events.py                     # Application events
├── integrated_analyzer.py        # Main analyzer with patterns
├── facies/
│   └── analyzer.py              # Facies analyzer (with @circuit_breaker, @retry)
├── patterns/
│   ├── dependency_injection.py  # DI implementation
│   ├── event_bus.py             # Event bus implementation
│   ├── circuit_breaker.py       # Circuit breaker implementation
│   └── retry.py                 # Retry implementation
└── examples/
    └── pattern_integration_demo.py  # Usage examples
```

## References

- Observer Pattern: `src/analysis/patterns/observer.py`
- Builder Pattern: `src/analysis/patterns/builder.py`
- Command Pattern: `src/analysis/patterns/command.py`
- Decorator Pattern: `src/analysis/decorators.py`

## Further Reading

- Dependency Injection: `src/analysis/patterns/dependency_injection.py`
- Event Bus: `src/analysis/patterns/event_bus.py`
- Circuit Breaker: `src/analysis/patterns/circuit_breaker.py`
- Retry Patterns: `src/analysis/patterns/retry.py`
"""
