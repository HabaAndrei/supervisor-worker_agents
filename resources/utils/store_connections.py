from logging_config import log_important_step
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from db_client import SYNC_DATABASE_URL
import asyncio

# Global connection pool instances (singleton pattern)
_checkpointer_instance: AsyncPostgresSaver | None = None
_checkpointer_context_manager = None
_checkpointer_lock = asyncio.Lock()


async def get_checkpointer() -> AsyncPostgresSaver:
    """
    Get or create the singleton AsyncPostgresSaver instance.

    This function implements connection pooling by reusing a single checkpointer instance
    across all requests instead of creating new connections per request.
    Thread-safe using asyncio.Lock.

    Note: AsyncPostgresSaver.from_conn_string() returns an async context manager.
    We must enter it and keep it alive for the lifetime of the application.
    """
    global _checkpointer_instance, _checkpointer_context_manager

    if _checkpointer_instance is None:
        async with _checkpointer_lock:
            # Double-check pattern to prevent race conditions
            if _checkpointer_instance is None:
                # Create the context manager
                _checkpointer_context_manager = AsyncPostgresSaver.from_conn_string(
                    SYNC_DATABASE_URL
                )
                # Enter the context manager and keep it alive
                _checkpointer_instance = (
                    await _checkpointer_context_manager.__aenter__()
                )
                # Setup the checkpointer (creates tables if needed)
                await _checkpointer_instance.setup()
                log_important_step(
                    "Initialized global AsyncPostgresSaver", "Connection pool ready"
                )

    return _checkpointer_instance