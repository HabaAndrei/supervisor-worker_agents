import asyncio
from uuid import uuid4
from langchain.chat_models import init_chat_model
from ..utils.state_declaration import GeneralChatAgentState
from langgraph.graph import StateGraph, START, END
from langchain.messages import (
    ToolMessage,
    AIMessage,
    SystemMessage,
    HumanMessage,
)
from .tools import call_story_generator_agent, call_character_creator_agent
from ..utils.store_connections import get_checkpointer
from langgraph.types import RetryPolicy
from llm_config.main import llm_config
from logging_config import log_error, log_message, log_important_step
from ..utils.handle_messages import fix_dangling_tool_calls
from ..utils.context_isolation import run_isolated

agent_config = llm_config["screenwriter_agent"]

llm = init_chat_model(agent_config["MODEL"], temperature=agent_config["TEMPERATURE"])

# Augment the LLM with tools
tools = [call_story_generator_agent, call_character_creator_agent]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = None


async def initialize_tools():
    """Initialize tools."""
    global llm_with_tools

    # Bind all tools to the LLM
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=True)

    return llm_with_tools


async def llm_call(state: GeneralChatAgentState):
    """LLM decides whether to call a tool or not"""

    if llm_with_tools is None:
        await initialize_tools()

    state_messages = state["messages"]

    updated_messages = await llm_with_tools.ainvoke(
        [SystemMessage(content=(agent_config["SYSTEM_PROMPT"]))]
        + state_messages,
        **agent_config.get("CONFIG", {}),
    )

    return {
        "messages": state_messages + [updated_messages],
    }


async def tool_node(state: dict):
    """
    Execute tools requested by the agent and return their results.
    """

    tool_messages = []

    # Log tool calls being executed in parallel
    tool_names = [tool_call["name"] for tool_call in state["messages"][-1].tool_calls]
    log_message(f"Screenwriter agent executing tools in parallel: {tool_names}")

    tool_calls = state["messages"][-1].tool_calls

    async def call_tool_individually(tool_call):
        tool_name = tool_call["name"]
        tool_args = tool_call["args"].copy()
        tool_call_id = tool_call["id"]

        log_important_step(
            f"Screenwriter agent executing tool: {tool_name}",
            f"Args: {list(tool_args.keys())}",
        )

        if tool_name in tools_by_name:
            # Execute local tool
            tool = tools_by_name[tool_name]
            tool_args["state"] = state

            try:
                # Execute the tool asynchronously and get the result
                tool_output = await tool.ainvoke(tool_args)
            except Exception as error:
                log_error(f"Screenwriter agent tool {tool_name} failed", error=error)
                return (tool_call_id, f"Error: Tool '{tool_name}' failed: {error}")

            log_message(f"Screenwriter agent tool {tool_name} completed successfully")
            return (tool_call_id, tool_output)
        else:
            log_error(
                "Screenwriter agent received unknown tool request",
                details=f"Tool: {tool_name}",
            )
            return (tool_call_id, f"Error: Unknown tool '{tool_name}'")

    # Process ALL tool calls in parallel
    tool_call_results = await asyncio.gather(
        *[call_tool_individually(tc) for tc in tool_calls]
    )

    for tool_call_id, tool_output in tool_call_results:
        if isinstance(tool_output, str) and tool_output.startswith("Error:"):
            tool_messages.append(
                ToolMessage(content=tool_output, tool_call_id=tool_call_id)
            )
        else:
            # Normal result
            tool_messages.append(
                ToolMessage(content=str(tool_output), tool_call_id=tool_call_id)
            )

    return {
        "messages": state["messages"] + tool_messages
    }


# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: GeneralChatAgentState):
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END


# Build workflow
main_graph = StateGraph(GeneralChatAgentState)

# Add nodes
main_graph.add_node(
    "llm_call", llm_call, retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0)
)
main_graph.add_node(
    "tool_node", tool_node, retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0)
)

# Add edges to connect nodes
main_graph.add_edge(START, "llm_call")
main_graph.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
main_graph.add_edge("tool_node", "llm_call")



async def compile_screenwriter_agent_graph(
    thread_id=None,
    human_message="Hey!",
):
    # Without an explicit thread id every caller would checkpoint into one
    # shared conversation, so fall back to a fresh unique thread
    if not thread_id:
        thread_id = uuid4().hex

    log_important_step("Screenwriter agent invoked", f"Thread id: {thread_id}")

    checkpointer = await get_checkpointer()
    compiled_main_graph = main_graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 1000}
    user_message = [HumanMessage(content=human_message)]

    async def execute_graph():
        state = await compiled_main_graph.aget_state(config)

        old_messages = fix_dangling_tool_calls(state[0].get("messages", []))

        return await compiled_main_graph.ainvoke(
            {
                "messages": old_messages + user_message,
                "thread_id": thread_id,
            },
            config,
            subgraphs=True,
            durability="async",
        )

    # This graph can run inside a parent agent's tool execution; run_isolated
    # severs the inherited runnable context so it checkpoints as its own
    # top-level conversation instead of as a per-call nested subgraph
    response = await run_isolated(execute_graph)

    response_messages = response.get("messages", [])

    final_content = response_messages[-1].content

    log_message(f"Screenwriter agent finished | Thread id: {thread_id}")

    return f"Thread id: {thread_id}\nContent: {final_content}"

# asyncio.run(compile_screenwriter_agent_graph(
#     thread_id="45--78-0--hcehj-cd-vc-df-v-ef-v-ef--vf",
#     human_message="Who are you?"
# ))

