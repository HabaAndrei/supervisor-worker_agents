from ...utils.state_declaration import GeneralChatAgentState
from langchain.tools import tool


@tool(description="")
async def call_director_agent(
    state: GeneralChatAgentState = None
):
    return True