from typing import Annotated

from langgraph.prebuilt import InjectedState

from ...story_generator_agent_resource.main import story_generator
from ...utils.state_declaration import GeneralChatAgentState
from langchain.tools import tool


@tool(
    description=(
        "Delegate story generation to the Story Generator agent. It creates the "
        "story concept: title, genre, setting, story summary, and main conflict. "
        "Call this before the Character Creator, since characters are built on "
        "top of the story concept. The Story Generator has no memory and only "
        "knows what you pass it, so 'task' must contain the complete assignment "
        "and all relevant context."
    )
)
async def call_story_generator_agent(
    task: str,
    state: Annotated[GeneralChatAgentState, InjectedState] = None,
):
    return await story_generator(task)
