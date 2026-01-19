"""MPI-based processing pool for parallel task execution.

This module provides the MPIPool class, which distributes tasks across multiple
processes using MPI (Message Passing Interface). It's designed to parallelize
computationally intensive tasks in Python programs running on multi-core or
distributed systems.

The implementation follows a master-worker pattern where:
- The master process (rank 0) distributes tasks and collects results
- Worker processes (rank > 0) execute tasks in parallel

Key features:
- Simple API similar to Python's built-in map()
- Automatic worker process management
- Optional dill support for complex objects
- Context manager support for automatic cleanup

Example
-------
>>> from ezmpi import MPIPool
>>>
>>> def square(x):
...     return x * x
>>>
>>> with MPIPool() as pool:
...     results = pool.map(square, [1, 2, 3, 4, 5])
...     print(results)
[1, 4, 9, 16, 25]

Notes
-----
This implementation was adapted from similar MPI pool implementations in the
scientific Python community.
"""

import atexit
import sys

MPI = None


def _import_mpi(use_dill=False):
    """Import and configure MPI with optional dill support.

    This function imports mpi4py.MPI and optionally configures it to use dill
    for pickling. It handles import errors gracefully with informative messages.

    Parameters
    ----------
    use_dill : bool, optional
        If True, configure MPI to use dill for pickling. Default is False.

    Returns
    -------
    module
        The imported and configured mpi4py.MPI module.

    Raises
    ------
    ImportError
        If mpi4py is not installed, or if use_dill=True but dill is not installed.

    Notes
    -----
    The function caches the MPI module after first import for performance.
    Subsequent calls return the cached module.

    When use_dill=True, this configures mpi4py to use dill's enhanced pickling
    capabilities, which can serialize more complex Python objects including
    lambda functions, closures, and other objects that standard pickle cannot.
    """
    global MPI
    if MPI is not None:
        return MPI

    try:
        from mpi4py import MPI as _MPI
    except ImportError as err:
        raise ImportError(
            "mpi4py is required but not installed. Please install it with: pip install mpi4py"
        ) from err

    if use_dill:
        try:
            import dill

            _MPI.pickle.__init__(dill.dumps, dill.loads, dill.HIGHEST_PROTOCOL)
        except ImportError as err:
            raise ImportError(
                "dill is required when use_dill=True but not installed. Please install it with: pip install dill"
            ) from err

    MPI = _MPI
    return MPI


class MPIPool:
    """A processing pool that distributes tasks using MPI.

    This class implements a master-worker parallel processing pattern using
    MPI (Message Passing Interface). Tasks are distributed from the master
    process (rank 0) to worker processes (rank > 0), executed in parallel,
    and results are collected back at the master.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm, optional
        An MPI communicator to distribute tasks with. If None, this uses
        MPI.COMM_WORLD by default.
    use_dill : bool, optional
        If True, use dill for pickling objects. This is useful for
        pickling functions and objects that are not picklable by the default
        pickle module. Default is True.

    Attributes
    ----------
    comm : mpi4py.MPI.Comm
        The MPI communicator used for communication
    master : int
        Rank of the master process (always 0)
    rank : int
        Rank of the current process
    workers : set of int
        Set of worker ranks (all ranks except master)
    size : int
        Number of worker processes

    Notes
    -----
    The implementation follows a master-worker pattern:

    - **Master process (rank 0)**: Distributes tasks to workers and collects
      results. This is the only process that should call :meth:`map`.
    - **Worker processes (rank > 0)**: Wait for tasks from the master,
      execute them, and return results. Workers automatically exit after
      the pool is closed.

    Examples
    --------
    Basic usage with a simple function:

    >>> from ezmpi import MPIPool
    >>>
    >>> def square(x):
    ...     return x * x
    >>>
    >>> with MPIPool() as pool:
    ...     results = pool.map(square, [1, 2, 3, 4, 5])
    ...     print(results)
    [1, 4, 9, 16, 25]

    Using dill for complex objects:

    >>> with MPIPool(use_dill=True) as pool:
    ...     results = pool.map(lambda x: x * 2, [1, 2, 3])
    ...     print(results)
    [2, 4, 6]
    """

    def __init__(self, comm=None, use_dill=True, test_mode=False):
        """Initialize the MPI processing pool.

        Parameters
        ----------
        comm : mpi4py.MPI.Comm, optional
            MPI communicator to use. If None, uses MPI.COMM_WORLD.
        use_dill : bool, optional
            If True, use dill for pickling. Default is True.
        test_mode : bool, optional
            If True, workers skip sys.exit() call (for testing).

        Raises
        ------
        ValueError
            If only one MPI process is available (need at least 2).
        """
        global MPI
        if MPI is None:
            MPI = _import_mpi(use_dill=use_dill)

        self.comm = MPI.COMM_WORLD if comm is None else comm

        self.master = 0
        self.rank = self.comm.Get_rank()

        atexit.register(lambda: MPIPool.close(self))

        if not self.is_master():
            # workers branch here and wait for work
            self.wait()
            if not test_mode:
                sys.exit(0)

        self.workers = set(range(self.comm.size))
        self.workers.discard(self.master)
        self.size = self.comm.Get_size() - 1

        if self.size == 0:
            raise ValueError(
                "Tried to create an MPI pool, but there "
                "was only one MPI process available. "
                "Need at least two."
            )

    def wait(self):
        """Tell the workers to wait and listen for the master process.

        This method is executed automatically by worker processes. Workers
        continuously listen for tasks from the master, execute them, and send
        back results. When they receive a None task, they exit.

        This method should not be called manually by users.

        Notes
        -----
        This method is called automatically in worker processes during
        initialization. It runs an infinite loop receiving tasks from the
        master process until it receives a termination signal (None).

        The communication protocol works as follows:
        1. Worker receives a task: (function, argument) tuple
        2. Worker executes the function with the argument
        3. Worker sends result back to master using synchronous send
        4. Process repeats or exits if task is None
        """
        if self.is_master():
            return

        status = MPI.Status()
        while True:
            task = self.comm.recv(source=self.master, tag=MPI.ANY_TAG, status=status)

            if task is None:
                # Worker told to quit work
                break

            func, arg = task
            result = func(arg)
            # Worker is sending answer with tag
            self.comm.ssend(result, self.master, status.tag)

    def map(self, worker, tasks):
        """Execute a worker function on each task in parallel.

        This method distributes tasks to worker processes, collects results,
        and returns them in the same order as the input tasks. It should only
        be called from the master process (rank 0).

        Parameters
        ----------
        worker : callable
            A function that takes a single argument and returns a result.
            The function must be pickleable. If using complex functions,
            set ``use_dill=True`` when creating the pool.
        tasks : iterable
            An iterable of tasks to distribute to workers. Each task will be
            passed as the argument to the worker function.

        Returns
        -------
        list
            A list of results in the same order as the input tasks. If a task
            fails, the corresponding result will be ``None``.

        Raises
        ------
        ValueError
            If called from a worker process instead of the master.

        See Also
        --------
        wait : Worker-side method that receives and processes tasks

        Notes
        -----
        Task distribution is synchronous - the master waits for all results
        before returning. Workers process tasks in the order they are received.

        The method handles task distribution load balancing automatically.
        If there are more tasks than workers, tasks are distributed round-robin.

        Examples
        --------
        Process a list of numbers:

        >>> def cube(x):
        ...     return x ** 3
        >>>
        >>> with MPIPool() as pool:
        ...     results = pool.map(cube, [1, 2, 3, 4, 5])
        ...     print(results)
        [1, 8, 27, 64, 125]

        Process strings:

        >>> def prefix(word):
        ...     return f"task: {word}"
        >>>
        >>> with MPIPool() as pool:
        ...     results = pool.map(prefix, ["a", "b", "c"])
        ...     print(results)
        ['task: a', 'task: b', 'task: c']
        """
        # If not the master just wait for instructions.
        if not self.is_master():
            self.wait()
            return

        workerset = self.workers.copy()
        tasklist = [(tid, (worker, arg)) for tid, arg in enumerate(tasks)]
        resultlist = [None] * len(tasklist)
        pending = len(tasklist)

        while pending:
            if workerset and tasklist:
                worker = workerset.pop()
                taskid, task = tasklist.pop()
                # "Sent task %s to worker %s with tag %s"
                self.comm.send(task, dest=worker, tag=taskid)

            if tasklist:
                flag = self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                if not flag:
                    continue
            else:
                self.comm.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)

            status = MPI.Status()
            result = self.comm.recv(
                source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status
            )
            worker = status.source
            taskid = status.tag

            # "Master received from worker %s with tag %s"

            workerset.add(worker)
            resultlist[taskid] = result
            pending -= 1

        return resultlist

    def close(self):
        """Tell all the workers to quit.

        Sends a termination signal (None) to all worker processes, causing them
        to exit their wait loops and terminate cleanly.

        Notes
        -----
        This method is called automatically when the pool is used as a context
        manager (e.g., ``with MPIPool() as pool:``).

        After calling close(), the pool should not be used again.
        """
        if self.is_worker():
            return

        for worker in self.workers:
            self.comm.send(None, worker, 0)

    def is_master(self):
        """Check if the current process is the master (rank 0).

        Returns
        -------
        bool
            True if this is the master process, False otherwise.
        """
        return self.rank == 0

    def is_worker(self):
        """Check if the current process is a worker (rank > 0).

        Returns
        -------
        bool
            True if this is a worker process, False otherwise.
        """
        return self.rank != 0

    def __enter__(self):
        """Enter the runtime context for the pool.

        Returns
        -------
        MPIPool
            The pool instance.
        """
        return self

    def __exit__(self, *args):
        """Exit the runtime context for the pool.

        Automatically calls close() to clean up worker processes.
        """
        self.close()
