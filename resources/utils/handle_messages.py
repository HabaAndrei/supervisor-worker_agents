from langchain.messages import ToolMessage, AIMessage
from logging_config import log_message


def fix_dangling_tool_calls(messages: list) -> list:
    """
    Detect AIMessages whose tool_calls never received a ToolMessage response
    (because the LangGraph execution was interrupted) and insert a placeholder
    ToolMessage for each unanswered call.

    Why this is needed:
        When a LangGraph graph stops mid-execution (crash, timeout, recursion
        limit, manual cancellation, or an unhandled exception in the tool node),
        the checkpoint may contain an AIMessage with tool_calls but no matching
        ToolMessage results. On the next invocation the LLM expects every
        tool_call to have a corresponding ToolMessage; without one the request
        fails with an "expected tool result" validation error.

    What it does:
        1. Scans all messages and collects every tool_call_id that already has
            a ToolMessage response.
        2. Walks the list again; when it finds an AIMessage with tool_calls
            whose ids are NOT in the answered set, it inserts a ToolMessage
            right after that AIMessage so the conversation stays valid.

    Real-world scenarios where this helps:

        Scenario 1 - Graph timeout / recursion limit
            The LLM called `call_flights_agent` (a long-running sub-agent tool)
            and the graph hit its recursion_limit or an asyncio timeout before
            the tool node could return. The checkpoint now ends with:
                AIMessage(tool_calls=[{id: "call_abc", name: "call_flights_agent", ...}])
            No ToolMessage follows. On resume this function adds:
                ToolMessage(content="This tool call never finished ...", tool_call_id="call_abc")
            so the LLM can gracefully continue.

        Scenario 2 - Multiple parallel tool calls, partial completion
            The LLM issued 3 parallel calls to `show_flights_to_user` (e.g. for
            March, April, and May). The first two completed and stored their
            ToolMessages, but the third was still running when the server
            restarted. The checkpoint contains:
                AIMessage(tool_calls=[{id: "call_1"}, {id: "call_2"}, {id: "call_3"}])
                ToolMessage(tool_call_id="call_1", content="Success!")
                ToolMessage(tool_call_id="call_2", content="Success!")
            This function detects that "call_3" has no response and inserts:
                ToolMessage(tool_call_id="call_3", content="This tool call never finished ...")

        Scenario 3 - Unhandled exception in tool_node
            A tool raises an unexpected error (e.g. network failure calling the
            flights API). Even though the tool_node has a RetryPolicy, after
            max retries are exhausted the graph stops. The last AIMessage's
            tool_calls remain unanswered. This function patches them on the
            next user message so the conversation can proceed.

    Args:
        messages: The full list of LangChain message objects loaded from the
            checkpoint (HumanMessage, AIMessage, ToolMessage, etc.).

    Returns:
        A new list with the same messages plus any missing ToolMessage
        placeholders inserted immediately after the AIMessage that made the
        unanswered call.
    """
    # Collect all tool_call_ids that already have a ToolMessage response
    answered_tool_call_ids = set()
    for msg in messages:
        if isinstance(msg, ToolMessage):
            answered_tool_call_ids.add(msg.tool_call_id)

    # Build corrected message list, inserting missing ToolMessages right after
    # the AIMessage that made the unanswered tool call
    corrected = []
    for msg in messages:
        corrected.append(msg)

        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            missing_calls = [
                tc for tc in msg.tool_calls if tc["id"] not in answered_tool_call_ids
            ]
            if missing_calls:
                log_message(
                    f"Repaired {len(missing_calls)} dangling tool call(s): {[tc['id'] for tc in missing_calls]}"
                )
            for tc in missing_calls:
                corrected.append(
                    ToolMessage(
                        content="This tool call never finished, the graph has been stopped.",
                        tool_call_id=tc["id"],
                    )
                )

    return corrected
