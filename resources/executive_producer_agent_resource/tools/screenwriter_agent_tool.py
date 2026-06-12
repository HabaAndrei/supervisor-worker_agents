from typing import Annotated
from uuid import uuid4

from langgraph.prebuilt import InjectedState

from ...screenwriter_agent_resource.main import compile_screenwriter_agent_graph
from ...utils.state_declaration import GeneralChatAgentState
from ...utils.thread_locks import get_thread_lock
from langchain.tools import tool


@tool(
    description=(
        "Assign narrative work to the Screenwriter agent. It develops the story "
        "concept and main characters (using its Story Generator and Character "
        "Creator) and returns a SCREENWRITER REPORT for your review. The "
        "Screenwriter must complete and have their work approved before the "
        "Director starts; never call this in parallel with the Director. Pass "
        "the complete assignment and any feedback or revision requests in "
        "'task'. Conversation continuity: omit 'thread_id' to start a NEW "
        "conversation with the Screenwriter; every report starts with a "
        "'Thread id' line - pass that exact value as 'thread_id' to RESUME "
        "that conversation, so the Screenwriter remembers all previous "
        "exchanges. Always resume the same conversation for revisions and "
        "follow-ups on the same work."
    )
)
async def call_screenwriter_agent(
    task: str,
    thread_id: str | None = None,
    state: Annotated[GeneralChatAgentState, InjectedState] = None,
):
    parent_thread_id = state.get("thread_id") if state else None
    expected_prefix = f"{parent_thread_id}-screenwriter"

    if thread_id:
        # Only conversations belonging to this project can be resumed - a
        # foreign or fabricated id would silently bleed another project's
        # context into the run
        if not thread_id.startswith(expected_prefix):
            return (
                f"Error: Invalid thread_id '{thread_id}'. Omit thread_id to "
                "start a new Screenwriter conversation, or pass the exact "
                "'Thread id' value returned by a previous call on this project."
            )
        child_thread_id = thread_id
    else:
        child_thread_id = f"{expected_prefix}-{uuid4().hex[:8]}"

    # Concurrent graph runs on one thread corrupt its persisted history, so
    # parallel calls to this tool serialize on the child thread
    async with get_thread_lock(child_thread_id):
        return await compile_screenwriter_agent_graph(
            thread_id=child_thread_id,
            human_message=task,
        )
