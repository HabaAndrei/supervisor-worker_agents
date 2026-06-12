from typing_extensions import NotRequired

from langchain_core.messages import AnyMessage
from langchain.agents import AgentState


class GeneralChatAgentState(AgentState):
    """
    Custom state class chat agent

    Attributes:
        messages (list[AnyMessage]): Conversation history
        thread_id: int | str
        subagent_threads: child conversation threads spawned by this agent,
            keyed by agent name, e.g. {"screenwriter": ["<thread_id>", ...]}.
            Persisted in the checkpoint so any run can be inspected later.
    """

    messages: list[AnyMessage]
    thread_id: int | str
    # NotRequired: the key only appears after the first tool round, and the
    # tools' InjectedState validation must accept the initial state without it
    subagent_threads: NotRequired[dict[str, list[str]]]