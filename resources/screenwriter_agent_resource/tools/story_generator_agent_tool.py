from ...utils.state_declaration import GeneralChatAgentState
from langchain.tools import tool


@tool(description="")
async def call_story_generator_agent(
    state: GeneralChatAgentState = None
):
    return True