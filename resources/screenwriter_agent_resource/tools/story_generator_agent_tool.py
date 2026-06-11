from ...utils.state_declaration import GeneralChatAgentState
from langchain.tools import tool


@tool(
    description=(
        "Delegate story generation to the Story Generator agent. It creates the "
        "story concept: title, genre, setting, story summary, and main conflict. "
        "Call this before the Character Creator, since characters are built on "
        "top of the story concept."
    )
)
async def call_story_generator_agent(
    state: GeneralChatAgentState = None
):
    return (
        "Error: The Story Generator agent is not implemented yet. Do not call "
        "this tool again. Continue the task yourself and clearly state in your "
        "report that the Story Generator was unavailable."
    )
