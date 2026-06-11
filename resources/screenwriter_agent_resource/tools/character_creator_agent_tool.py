from ...character_creator_agent_resource.main import character_creator
from ...utils.state_declaration import GeneralChatAgentState
from langchain.tools import tool


@tool(
    description=(
        "Delegate character creation to the Character Creator agent. It creates "
        "the character profiles, motivations, and relationships for an existing "
        "story concept. Only call this after the Story Generator has produced "
        "the story concept. The Character Creator has no memory and only knows "
        "what you pass it, so 'task' must contain the full story concept and "
        "the assignment."
    )
)
async def call_character_creator_agent(
    task: str,
    state: GeneralChatAgentState = None,
):
    return await character_creator(task)
