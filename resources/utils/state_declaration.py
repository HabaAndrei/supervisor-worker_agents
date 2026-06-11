from langchain_core.messages import AnyMessage
from langchain.agents import AgentState


class GeneralChatAgentState(AgentState):
    """
    Custom state class chat agent

    Attributes:
        messages (list[AnyMessage]): Conversation history
        thread_id: int | str
    """

    messages: list[AnyMessage]
    thread_id: int | str