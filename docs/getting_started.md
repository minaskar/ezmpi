# Getting Started with EZMPI

This tutorial will guide you through installing and using EZMPI to parallelize your Python code.

## Prerequisites

Before you begin, make sure you have:

1. **Python 3.8 or higher** installed
2. An **MPI implementation** (OpenMPI, MPICH, or similar)
3. Basic familiarity with Python and the command line

## Installation

### Step 1: Install MPI

If you don't have MPI installed, install it for your system:

**macOS (using Homebrew):**
```bash
brew install open-mpi
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install openmpi-bin libopenmpi-dev
```

**CentOS/RHEL:**
```bash
sudo yum install openmpi openmpi-devel
```

Verify MPI is installed:
```bash
mpiexec --version
```

### Step 2: Install EZMPI

Install EZMPI from PyPI:

```bash
pip install ezmpi
```

Or install from source:

```bash
git clone https://github.com/minaskar/ezmpi.git
cd ezmpi
pip install -e .
```

### Step 3: Verify Installation

Create a simple test script `test_ezmpi.py`:

```python
# test_ezmpi.py
from ezmpi import MPIPool

def square(x):
    return x * x

if __name__ == "__main__":
    with MPIPool() as pool:
        if pool.is_master():
            numbers = [1, 2, 3, 4, 5]
            results = pool.map(square, numbers)
            print(f"Results: {results}")
```

Run it with MPI:

```bash
mpiexec -n 2 python test_ezmpi.py
```

You should see:
```
Results: [1, 4, 9, 16, 25]
```

## Basic Usage

### The Simplest Example

```python
from ezmpi import MPIPool

def add_one(x):
    return x + 1

with MPIPool() as pool:
    results = pool.map(add_one, [1, 2, 3, 4, 5])
    print(results)  # [2, 3, 4, 5, 6]
```

Save this as `simple.py` and run:

```bash
mpiexec -n 4 python simple.py
```

### Understanding the API

The `MPIPool` class has a simple API:

- **Initialization**: `MPIPool(comm=None, use_dill=True)`
- **Main method**: `pool.map(worker, tasks)`
- **Context manager**: `with MPIPool() as pool:`
- **Check role**: `pool.is_master()` or `pool.is_worker()`

### Working with Different Data Types

EZMPI can handle any picklable Python objects:

```python
from ezmpi import MPIPool

def process_data(item):
    name, age, score = item
    return {
        'name': name.upper(),
        'category': 'adult' if age >= 18 else 'minor',
        'passed': score >= 60
    }

data = [
    ('Alice', 25, 85),
    ('Bob', 17, 72),
    ('Charlie', 30, 58),
    ('Diana', 22, 91)
]

with MPIPool() as pool:
    results = pool.map(process_data, data)
    print(results)
```

## Common Patterns

### Pattern 1: CPU-Bound Computations

Best use case for EZMPI - expensive computations:

```python
from ezmpi import MPIPool
import math

def is_prime(n):
    """Check if a number is prime (CPU-intensive)."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

# Check primes in a range
numbers = list(range(2, 10000))

with MPIPool() as pool:
    # This will distribute the work across all workers
    results = [n for n, is_p in zip(numbers, pool.map(is_prime, numbers)) if is_p]
    print(f"Found {len(results)} primes between 2 and 10000")
    print(f"Largest prime: {results[-1]}")
```

### Pattern 2: Processing Files

```python
from ezmpi import MPIPool
import json

def process_file(filename):
    """Process a single file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Do some processing
    result = {
        'filename': filename,
        'record_count': len(data),
        'avg_value': sum(d['value'] for d in data) / len(data)
    }
    return result

# List of files to process
files = ['data1.json', 'data2.json', 'data3.json', 'data4.json']

with MPIPool() as pool:
    results = pool.map(process_file, files)
    
    for result in results:
        print(f"{result['filename']}: {result['record_count']} records, avg={result['avg_value']:.2f}")
```

### Pattern 3: Parameter Sweeps

```python
from ezmpi import MPIPool
import numpy as np

def run_simulation(params):
    """Run a simulation with given parameters."""
    learning_rate, batch_size, epochs = params
    
    # Simulate training (in real code, you'd train a model)
    np.random.seed(hash(params) % 2**32)
    loss = np.random.exponential(scale=1.0/learning_rate)
    accuracy = 1.0 - loss + np.random.normal(0, 0.1)
    
    return {
        'params': params,
        'loss': loss,
        'accuracy': max(0, min(1, accuracy))
    }

# Define parameter grid
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [32, 64, 128]
epochs = [10, 50, 100]

# Create all combinations
param_combinations = [(lr, bs, ep) for lr in learning_rates 
                                      for bs in batch_sizes 
                                      for ep in epochs]

with MPIPool() as pool:
    results = pool.map(run_simulation, param_combinations)
    
    # Find best parameters
    best = min(results, key=lambda x: x['loss'])
    print(f"Best parameters: {best['params']}")
    print(f"Best loss: {best['loss']:.4f}")
    print(f"Best accuracy: {best['accuracy']:.4f}")
```

## Using dill for Complex Functions

Standard pickle can't handle lambda functions or locally defined functions. Use `use_dill=True`:

```python
from ezmpi import MPIPool

def create_multiplier(factor):
    """Create a multiplier function (closure)."""
    def multiply(x):
        return x * factor
    return multiply

# Create multiplier functions
multipliers = [create_multiplier(i) for i in range(1, 6)]

with MPIPool(use_dill=True) as pool:
    results = []
    for multiplier in multipliers:
        result = pool.map(multiplier, [1, 2, 3])
        results.append(result)
    
    print(results)
    # [[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12], [5, 10, 15]]
```

## Best Practices

### 1. Use Context Managers

Always use the context manager for automatic cleanup:

```python
# Good
with MPIPool() as pool:
    results = pool.map(worker, tasks)

# Avoid (manual cleanup needed)
pool = MPIPool()
results = pool.map(worker, tasks)
pool.close()  # Don't forget!
```

### 2. Choose Appropriate Task Granularity

- **Too small**: Communication overhead dominates
- **Too large**: Poor load balancing
- **Just right**: 0.1-10 seconds per task

### 3. Use dill When Needed

```python
# Simple function - no dill needed
def simple(x):
    return x * 2

with MPIPool() as pool:
    results = pool.map(simple, tasks)

# Complex function - use dill
with MPIPool(use_dill=True) as pool:
    # closures, lambdas, nested functions
    results = pool.map(complex_function, tasks)
```

### 4. Handle Errors Gracefully

```python
from ezmpi import MPIPool

def safe_worker(x):
    """Worker that handles errors."""
    try:
        return risky_computation(x)
    except Exception as e:
        print(f"Error processing {x}: {e}")
        return None  # Return None on error

with MPIPool() as pool:
    results = pool.map(safe_worker, data)
    # Filter out failed computations
    successful_results = [r for r in results if r is not None]
```

## Troubleshooting

### "only one MPI process available"

You need at least 2 MPI processes:

```bash
# Wrong
mpiexec -n 1 python script.py

# Correct
mpiexec -n 2 python script.py
```

### "mpi4py is required"

Install mpi4py:

```bash
pip install mpi4py
```

### "dill is required when use_dill=True"

Install dill:

```bash
pip install dill
```

### Hanging or Not Terminating

Make sure you're using a context manager or calling `close()`:

```python
# Good
with MPIPool() as pool:
    results = pool.map(worker, tasks)

# Good
pool = MPIPool()
try:
    results = pool.map(worker, tasks)
finally:
    pool.close()
```

## Next Steps

- Read the [API Reference](api.md) for detailed documentation
- Check out the [Architecture](architecture.md) to understand how EZMPI works
- See [Contributing](contributing.md) if you want to help improve EZMPI

---

**Need help?** Open an issue on [GitHub](https://github.com/minaskar/ezmpi/issues)!
