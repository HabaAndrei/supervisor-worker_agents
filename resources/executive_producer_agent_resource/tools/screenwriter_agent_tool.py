from typing import Annotated

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
        "'task'. The Screenwriter remembers previous exchanges on this project."
    )
)
async def call_screenwriter_agent(
    task: str,
    state: Annotated[GeneralChatAgentState, InjectedState] = None,
):
    parent_thread_id = state.get("thread_id") if state else None
    child_thread_id = f"{parent_thread_id}-screenwriter"

    # Concurrent graph runs on one thread corrupt its persisted history, so
    # parallel calls to this tool serialize on the child thread
    async with get_thread_lock(child_thread_id):
        return await compile_screenwriter_agent_graph(
            thread_id=child_thread_id,
            human_message=task,
        )
