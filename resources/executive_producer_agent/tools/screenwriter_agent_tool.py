from ..state_declaration import ChatAgentState
from langchain.tools import tool


@tool(description="")
async def call_screenwriter_agent(
    state: ChatAgentState = None
):
    return True