import re

# Subagent reports start with a "Thread id: <id>" line; that line is the only
# place a child thread id surfaces, so it is the contract this module parses
THREAD_ID_PATTERN = re.compile(r"^Thread id: (\S+)", re.MULTILINE)

# Maps each delegation tool to the agent it spawns, so state keys read as
# agent names instead of tool names
AGENT_NAME_BY_TOOL = {
    "call_screenwriter_agent": "screenwriter",
    "call_director_agent": "director",
    "call_story_generator_agent": "story_generator",
    "call_character_creator_agent": "character_creator",
    "call_scene_planner_agent": "scene_planner",
}


def extract_thread_id(tool_output) -> str | None:
    """Return the child thread id from a subagent report, if it has one.

    Stateless agents (e.g. Story Generator) return plain content with no
    thread id, and errored delegations never spawned a conversation - both
    return None.
    """
    text = str(tool_output or "")
    if text.startswith("Error:"):
        return None
    match = THREAD_ID_PATTERN.search(text)
    return match.group(1) if match else None


def merge_subagent_threads(current, tool_calls, tool_call_results) -> dict:
    """Fold this round of tool results into the subagent_threads state value.

    Args:
        current: existing subagent_threads dict from state (may be None).
        tool_calls: the AIMessage tool_calls that were executed.
        tool_call_results: list of (tool_call_id, tool_output) pairs.

    Returns a NEW dict (state values must not be mutated in place) mapping
    agent name -> ordered, de-duplicated list of child thread ids. Resumed
    conversations reuse their id, so re-delegations add nothing new.
    """
    merged = {name: list(threads) for name, threads in (current or {}).items()}
    tool_name_by_id = {tc["id"]: tc["name"] for tc in tool_calls}

    for tool_call_id, tool_output in tool_call_results:
        thread_id = extract_thread_id(tool_output)
        if not thread_id:
            continue
        tool_name = tool_name_by_id.get(tool_call_id, "unknown")
        agent_name = AGENT_NAME_BY_TOOL.get(tool_name, tool_name)
        threads = merged.setdefault(agent_name, [])
        if thread_id not in threads:
            threads.append(thread_id)

    return merged
