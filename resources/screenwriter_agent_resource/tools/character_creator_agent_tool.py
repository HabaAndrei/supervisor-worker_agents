from ...utils.state_declaration import GeneralChatAgentState
from langchain.tools import tool


@tool(
    description=(
        "Delegate character creation to the Character Creator agent. It creates "
        "the main characters for an existing story concept. Only call this after "
        "the Story Generator has produced the story concept."
    )
)
async def call_character_creator_agent(
    state: GeneralChatAgentState = None
):
    return (
        "Error: The Character Creator agent is not implemented yet. Do not call "
        "this tool again. Continue the task yourself and clearly state in your "
        "report that the Character Creator was unavailable."
    )
