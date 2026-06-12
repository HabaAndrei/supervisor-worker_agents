from typing import Annotated
from uuid import uuid4

from langgraph.prebuilt import InjectedState

from ...director_agent_resource.main import compile_director_agent_graph
from ...utils.state_declaration import GeneralChatAgentState
from ...utils.thread_locks import get_thread_lock
from langchain.tools import tool


@tool(
    description=(
        "Assign production planning to the Director agent. It converts the "
        "approved story into a structured production vision (using its Scene "
        "Planner) and returns a DIRECTOR REPORT for your review. Only call this "
        "after the Screenwriter's work has been reviewed and approved; never "
        "call it in parallel with the Screenwriter. Conversation continuity: "
        "omit 'thread_id' to start a NEW conversation - in that case 'task' "
        "must contain the FULL approved story and characters. Every report "
        "starts with a 'Thread id' line - pass that exact value as 'thread_id' "
        "to RESUME that conversation, so the Director remembers all previous "
        "exchanges. Always resume the same conversation for revisions and "
        "follow-ups on the same work."
    )
)
async def call_director_agent(
    task: str,
    thread_id: str | None = None,
    state: Annotated[GeneralChatAgentState, InjectedState] = None,
):
    parent_thread_id = state.get("thread_id") if state else None
    expected_prefix = f"{parent_thread_id}-director"

    if thread_id:
        # Only conversations belonging to this project can be resumed - a
        # foreign or fabricated id would silently bleed another project's
        # context into the run
        if not thread_id.startswith(expected_prefix):
            return (
                f"Error: Invalid thread_id '{thread_id}'. Omit thread_id to "
                "start a new Director conversation, or pass the exact "
                "'Thread id' value returned by a previous call on this project."
            )
        child_thread_id = thread_id
    else:
        child_thread_id = f"{expected_prefix}-{uuid4().hex[:8]}"

    # Concurrent graph runs on one thread corrupt its persisted history, so
    # parallel calls to this tool serialize on the child thread
    async with get_thread_lock(child_thread_id):
        return await compile_director_agent_graph(
            thread_id=child_thread_id,
            human_message=task,
        )
