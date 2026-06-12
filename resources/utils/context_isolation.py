import asyncio

from langchain_core.runnables.config import var_child_runnable_config


async def run_isolated(coro_fn):
    """
    Run an async callable outside any inherited LangChain runnable context.

    When a graph agent is invoked from inside another agent's tool execution,
    LangChain propagates the parent run's config (checkpoint namespace, pregel
    task id, callbacks) through a contextvar, which makes the child graph
    checkpoint as a nested subgraph under a per-call namespace instead of as
    its own top-level conversation. Resetting the contextvar inside a fresh
    task severs that inheritance without affecting the caller's context.
    """

    async def _runner():
        var_child_runnable_config.set(None)
        return await coro_fn()

    # create_task copies the current contextvars context, so the reset above
    # stays local to this run
    return await asyncio.create_task(_runner())
