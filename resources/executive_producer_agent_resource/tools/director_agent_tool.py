from ...director_agent_resource.main import compile_director_agent_graph
from ...utils.state_declaration import GeneralChatAgentState
from langchain.tools import tool


@tool(
    description=(
        "Assign production planning to the Director agent. It converts the "
        "approved story into a structured production vision (using its Scene "
        "Planner) and returns a DIRECTOR REPORT for your review. Only call this "
        "after the Screenwriter's work has been reviewed and approved; never "
        "call it in parallel with the Screenwriter. The Director only knows "
        "what you pass it, so 'task' must contain the FULL approved story and "
        "characters. The Director remembers previous exchanges on this project."
    )
)
async def call_director_agent(
    task: str,
    state: GeneralChatAgentState = None,
):
    parent_thread_id = state.get("thread_id") if state else None
    child_thread_id = f"{parent_thread_id}-director"

    return await compile_director_agent_graph(
        thread_id=child_thread_id,
        human_message=task,
    )
