import asyncio
from collections import defaultdict

# One lock per checkpointer thread id (process-local)
_thread_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


def get_thread_lock(thread_id: str) -> asyncio.Lock:
    """
    Lock guarding graph executions on a single checkpointer thread.

    Two graph runs on the same thread id race on read-modify-write of the
    persisted message history (last writer wins), so callers must serialize
    runs per thread id while still allowing different threads in parallel.
    """
    return _thread_locks[thread_id]
