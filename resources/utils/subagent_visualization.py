"""Visualization layer for the subagent hierarchy.

Every graph agent checkpoints the thread ids of the subagents it spawned in
its `subagent_threads` state value (see subagent_tracking.py). This module
reads that value back from the checkpointer and rebuilds the entire
delegation tree of a run from state alone - no log scraping required.

This is a library only: the user-facing entry point is
visualize_subagents(main_thread_id) in the root main.py.

Programmatic use:
    threads = await get_subagent_threads(thread_id)   # {"screenwriter": [...]}
    tree = await build_subagent_tree(thread_id)       # full nested dict
    print(render_subagent_tree(tree))                 # ASCII tree
    report = await build_run_transcript(thread_id)    # full interaction log (markdown)
"""

from .store_connections import get_checkpointer
from .subagent_tracking import AGENT_NAME_BY_TOOL

# Agents that run as checkpointed graphs vs. single stateless LLM calls -
# stateless agents have no thread of their own, so they only appear in the
# tree through their parent's tool-call history
STATELESS_AGENTS = {"story_generator", "character_creator", "scene_planner"}

ROOT_AGENT_NAME = "executive_producer"


async def get_state_values(thread_id: str) -> dict | None:
    """Return the latest checkpointed state values for a thread, or None."""
    checkpointer = await get_checkpointer()
    checkpoint_tuple = await checkpointer.aget_tuple(
        {"configurable": {"thread_id": str(thread_id)}}
    )
    if checkpoint_tuple is None:
        return None
    return checkpoint_tuple.checkpoint.get("channel_values", {})


async def get_subagent_threads(thread_id: str) -> dict[str, list[str]]:
    """Easy retrieval of an agent's child thread ids straight from state.

    Returns e.g. {"screenwriter": ["<id>"], "director": ["<id>"]} for an
    Executive Producer thread, or {} for threads with no stateful children
    (runs recorded before tracking existed also return {}).
    """
    values = await get_state_values(thread_id)
    if not values:
        return {}
    return values.get("subagent_threads") or {}


def infer_agent_name(thread_id: str) -> str:
    """Derive the agent a thread belongs to from the child-id naming scheme
    '{parent}-{agent}-{hex}'; ids without a marker are root producer threads."""
    inferred = ROOT_AGENT_NAME
    last_position = -1
    for agent_name in AGENT_NAME_BY_TOOL.values():
        position = str(thread_id).rfind(f"-{agent_name}-")
        if position > last_position:
            last_position = position
            inferred = agent_name
    return inferred


def count_delegations(messages) -> dict[str, int]:
    """Count how many times each subagent was called, from the AI messages'
    tool calls in a thread's history."""
    counts: dict[str, int] = {}
    for message in messages or []:
        for tool_call in getattr(message, "tool_calls", None) or []:
            agent_name = AGENT_NAME_BY_TOOL.get(tool_call["name"], tool_call["name"])
            counts[agent_name] = counts.get(agent_name, 0) + 1
    return counts


async def build_subagent_tree(thread_id: str, agent_name: str | None = None) -> dict:
    """Rebuild the full delegation tree of a run from checkpointed state.

    Recurses through `subagent_threads` for stateful children (their threads
    are checkpointed too) and lists stateless agents from the delegation
    counts in the message history.
    """
    values = await get_state_values(thread_id)

    node = {
        "agent": agent_name or infer_agent_name(thread_id),
        "thread_id": str(thread_id),
        "checkpoint_found": values is not None,
        "message_count": len((values or {}).get("messages") or []),
        "delegation_counts": count_delegations((values or {}).get("messages")),
        "children": [],
    }
    if values is None:
        return node

    subagent_threads = values.get("subagent_threads") or {}

    # Stateful children: one subtree per recorded child thread
    for child_agent, child_thread_ids in subagent_threads.items():
        for child_thread_id in child_thread_ids:
            node["children"].append(
                await build_subagent_tree(child_thread_id, agent_name=child_agent)
            )

    # Stateless children: no thread to recurse into, shown with call counts
    for child_agent, calls in node["delegation_counts"].items():
        if child_agent in STATELESS_AGENTS:
            node["children"].append(
                {
                    "agent": child_agent,
                    "thread_id": None,
                    "stateless": True,
                    "calls": calls,
                    "children": [],
                }
            )

    return node


def render_subagent_tree(node: dict) -> str:
    """Render a tree from build_subagent_tree as ASCII, one agent per line."""
    if node.get("stateless"):
        calls = node["calls"]
        label = f"{node['agent']}  (stateless, {calls} call{'s' if calls != 1 else ''})"
    elif not node.get("checkpoint_found"):
        label = f"{node['agent']}  [thread: {node['thread_id']}]  (no checkpoint found)"
    else:
        label = (
            f"{node['agent']}  [thread: {node['thread_id']}]  "
            f"({node['message_count']} messages)"
        )

    lines = [label]
    children = node.get("children") or []
    for index, child in enumerate(children):
        is_last = index == len(children) - 1
        branch = "└── " if is_last else "├── "
        continuation = "    " if is_last else "│   "
        child_lines = render_subagent_tree(child).split("\n")
        lines.append(branch + child_lines[0])
        lines.extend(continuation + line for line in child_lines[1:])
    return "\n".join(lines)


def _stateful_nodes(node) -> list[dict]:
    """Flatten the tree to its checkpointed agents, parent before children.

    Stateless leaf agents are skipped: their full interaction already appears
    inside the parent's transcript as a tool call + tool result.
    """
    nodes = [node]
    for child in node.get("children") or []:
        if not child.get("stateless"):
            nodes.extend(_stateful_nodes(child))
    return nodes


def render_thread_transcript(agent_name: str, thread_id: str, messages) -> str:
    """Render one thread's full message history as readable markdown."""
    lines = [
        f"## {agent_name} — thread `{thread_id}` ({len(messages)} messages)",
        "",
    ]

    # Tool results only carry a tool_call_id; recover the tool name from the
    # AI message that requested the call
    tool_name_by_call_id = {}
    for message in messages:
        for tool_call in getattr(message, "tool_calls", None) or []:
            tool_name_by_call_id[tool_call["id"]] = tool_call["name"]

    for index, message in enumerate(messages, 1):
        role = getattr(message, "type", type(message).__name__)
        content = str(message.content or "").strip()
        tool_calls = getattr(message, "tool_calls", None) or []

        if role == "human":
            lines.append(f"### [{index}] Task given to {agent_name}")
            lines.extend(["", content, ""])
        elif role == "ai" and tool_calls:
            called = ", ".join(tc["name"] for tc in tool_calls)
            lines.append(f"### [{index}] {agent_name} delegates → {called}")
            if content:
                lines.extend(["", content])
            for tool_call in tool_calls:
                arguments = tool_call.get("args") or {}
                lines.extend(["", f"**{tool_call['name']}**"])
                for key, value in arguments.items():
                    if key == "task":
                        lines.extend(["", "task:", "", "```", str(value), "```"])
                    else:
                        lines.append(f"- {key}: `{value}`")
            lines.append("")
        elif role == "ai":
            lines.append(f"### [{index}] {agent_name} final answer")
            lines.extend(["", content, ""])
        elif role == "tool":
            tool_name = tool_name_by_call_id.get(
                getattr(message, "tool_call_id", None), "tool"
            )
            lines.append(f"### [{index}] Result from {tool_name}")
            lines.extend(["", content, ""])
        else:
            lines.append(f"### [{index}] {role}")
            lines.extend(["", content, ""])

    return "\n".join(lines)


async def build_run_transcript(main_thread_id: str) -> str:
    """Build the full interaction report of a run as one markdown document.

    Contains the delegation tree followed by the complete, untruncated
    message history of every checkpointed agent in the hierarchy - every
    task, every tool call with its arguments, every report - so the whole
    chain of reasoning between agents can be read top to bottom.
    """
    tree = await build_subagent_tree(main_thread_id)

    sections = [
        f"# Subagent run `{main_thread_id}`",
        "",
        "## Delegation tree",
        "",
        "```",
        render_subagent_tree(tree),
        "```",
        "",
    ]

    for node in _stateful_nodes(tree):
        values = await get_state_values(node["thread_id"]) or {}
        sections.append(
            render_thread_transcript(
                node["agent"], node["thread_id"], values.get("messages") or []
            )
        )
        sections.append("")

    return "\n".join(sections)
