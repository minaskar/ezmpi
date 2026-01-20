# Getting Started with ezMPI

ezMPI is a simple and efficient MPI processing pool for Python that helps you parallelize computationally intensive tasks across multiple processes.

## Installation

### Prerequisites

- Python 3.8 or higher
- MPI implementation (OpenMPI, MPICH, or MS-MPI)
- mpi4py

### Install from PyPI

```bash
pip install ezmpi
```

### Install from Source

```bash
git clone https://github.com/minaskar/ezmpi.git
cd ezmpi
pip install -e .
```

### With Optional Dependencies

For enhanced pickle support with complex objects:

```bash
pip install ezmpi[dill]
```

Or from source:

```bash
pip install -e ".[dill]"
```

## Basic Usage

### Simple Example

```python
from ezmpi import MPIPool

def square(x):
    """Simple function to parallelize"""
    return x * x

# Run with: mpiexec -n 4 python script.py
if __name__ == "__main__":
    with MPIPool() as pool:
        if pool.is_master():
            # Master coordinates work
            tasks = [1, 2, 3, 4, 5]
            results = pool.map(square, tasks)
            print(f"Results: {results}")
            # Output: Results: [1, 4, 9, 16, 25]
```

### CPU-Intensive Work

```python
from ezmpi import MPIPool
import time

def heavy_computation(x):
    """CPU-bound task"""
    result = 0
    for i in range(1000000):
        result += i * x
    return result

with MPIPool() as pool:
    if pool.is_master():
        tasks = [1, 2, 3, 4, 5, 6, 7, 8]
        start = time.time()
        results = pool.map(heavy_computation, tasks)
        duration = time.time() - start
        print(f"Computed {len(results)} tasks in {duration:.2f}s")
```

### Working with Complex Objects

```python
from ezmpi import MPIPool

class DataProcessor:
    def __init__(self, multiplier):
        self.multiplier = multiplier
    
    def process(self, data):
        return [x * self.multiplier for x in data]

# Use dill for pickling complex objects
with MPIPool(use_dill=True) as pool:
    if pool.is_master():
        processor = DataProcessor(3)
        tasks = [[1, 2], [3, 4], [5, 6]]
        results = pool.map(processor.process, tasks)
        print(results)
```

## Command Line Examples

Run with different numbers of processes:

```bash
# Use 4 processes
mpiexec -n 4 python your_script.py

# Use 8 processes (oversubscribe if needed)
mpiexec --oversubscribe -n 8 python your_script.py
```

## Performance Tips

1. **Task Granularity**: Ensure tasks are large enough to outweigh MPI communication overhead
   ```python
   # Good: Large tasks
   def compute(x):
       return sum(i * i for i in range(x, x + 10000))
   
   # Not ideal: Very small tasks
   def small_task(x):
       return x + 1
   ```

2. **Data Size**: Minimize data transfer between master and workers
   ```python
   # Pass small objects, generate large data on workers
   def generate_and_process(seed):
       data = generate_large_data(seed)  # Generate on worker
         return compute(data)  # Process on worker
   ```

3. **Batch Operations**: Use `map()` for batches of similar tasks
   ```python
   results = pool.map(process_item, large_list_of_items)
   ```

## Troubleshooting

### ImportError: No module named 'mpi4py'

Install mpi4py:
```bash
pip install mpi4py
```

### Only one MPI process available

ezMPI requires at least 2 MPI processes:
```bash
# Wrong: Only 1 process
mpiexec -n 1 python script.py

# Correct: 2+ processes
mpiexec -n 2 python script.py
```

### Performance not improving

- Ensure tasks are CPU-bound
- Check task size vs communication overhead
- Verify you're using enough processes
- Profile your code to find bottlenecks

## Learn More

- [API Reference](api.md) - Complete API documentation
- [Architecture](architecture.md) - Technical implementation details
- [Contributing](contributing.md) - How to contribute to ezMPI

## Need Help?

- Open an issue on [GitHub](https://github.com/minaskar/ezmpi/issues)
- Check existing issues for similar problems
- Review the [test suite](../tests/) for usage examples
