from logging_config import log_important_step
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from db_client import SYNC_DATABASE_URL
import asyncio

# Global connection pool instances (singleton pattern)
_checkpointer_instance: AsyncPostgresSaver | None = None
_connection_pool: AsyncConnectionPool | None = None
_checkpointer_lock = asyncio.Lock()


async def get_checkpointer() -> AsyncPostgresSaver:
    """
    Get or create the singleton AsyncPostgresSaver instance.

    Backed by an AsyncConnectionPool so that the parent and nested child graphs
    checkpoint over separate connections instead of contending on a single one,
    and so that a dropped connection is replaced automatically by the pool.
    Thread-safe using asyncio.Lock.
    """
    global _checkpointer_instance, _connection_pool

    if _checkpointer_instance is None:
        async with _checkpointer_lock:
            # Double-check pattern to prevent race conditions
            if _checkpointer_instance is None:
                _connection_pool = AsyncConnectionPool(
                    SYNC_DATABASE_URL,
                    open=False,
                    # Validate connections at checkout so dead ones (server
                    # restart, dropped session) are replaced instead of handed
                    # to a graph run
                    check=AsyncConnectionPool.check_connection,
                    kwargs={
                        "autocommit": True,
                        "prepare_threshold": 0,
                        "row_factory": dict_row,
                    },
                )
                await _connection_pool.open()

                checkpointer = AsyncPostgresSaver(_connection_pool)
                # Setup the checkpointer (creates tables if needed)
                await checkpointer.setup()

                # Publish only after setup so other coroutines never see a
                # half-initialized instance
                _checkpointer_instance = checkpointer
                log_important_step(
                    "Initialized global AsyncPostgresSaver", "Connection pool ready"
                )

    return _checkpointer_instance
