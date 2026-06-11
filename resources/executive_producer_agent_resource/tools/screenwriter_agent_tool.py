from ...utils.state_declaration import GeneralChatAgentState
from langchain.tools import tool


@tool(
    description=(
        "Assign narrative work to the Screenwriter agent. It develops the story "
        "concept and main characters and returns a SCREENWRITER REPORT for your "
        "review. The Screenwriter must complete and have their work approved "
        "before the Director starts; never call this in parallel with the "
        "Director."
    )
)
async def call_screenwriter_agent(
    state: GeneralChatAgentState = None
):
    return (
        "Error: The Screenwriter agent is not implemented yet. Do not call this "
        "tool again. Continue the task yourself and clearly state in your final "
        "response that the Screenwriter was unavailable."
    )
