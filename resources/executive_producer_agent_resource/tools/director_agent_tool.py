from ...utils.state_declaration import GeneralChatAgentState
from langchain.tools import tool


@tool(
    description=(
        "Assign production planning to the Director agent. It turns the approved "
        "story and characters into a production plan (scene planning and visual "
        "direction). Only call this after the Screenwriter's work has been "
        "reviewed and approved; never call it in parallel with the Screenwriter."
    )
)
async def call_director_agent(
    state: GeneralChatAgentState = None
):
    return (
        "Error: The Director agent is not implemented yet. Do not call this "
        "tool again. Continue the task yourself and clearly state in your final "
        "response that the Director was unavailable."
    )
