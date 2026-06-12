import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()


async def run_main_agent(human_message: str, thread_id: str | None = None) -> str:
    """Run the Executive Producer, the main agent of the production system.

    It coordinates the whole hierarchy (Screenwriter, Director and their
    subagents) and returns the final result, prefixed with the main thread
    id of the run. Pass that same thread_id back to continue the
    conversation, or to visualize_subagents() to inspect the run.
    """
    from resources.executive_producer_agent_resource.main import (
        compile_executive_producer_agent_graph,
    )

    result = await compile_executive_producer_agent_graph(
        thread_id=thread_id,
        human_message=human_message,
    )
    print(result)
    return result


async def visualize_subagents(main_thread_id: str) -> str:
    """Print the full subagent delegation tree of a run.

    The only entry point for visualization: pass the MAIN thread id (the one
    run_main_agent was invoked with) and the whole hierarchy - child agent
    threads and stateless leaf calls - is rebuilt from checkpointed state
    and printed as an ASCII tree.
    """
    from resources.utils.subagent_visualization import (
        build_subagent_tree,
        render_subagent_tree,
    )

    tree = await build_subagent_tree(main_thread_id)
    rendering = render_subagent_tree(tree)
    print(rendering)
    return rendering


def main():
    print(
        "Usage:\n"
        '  python main.py run "<message>" [thread_id]   '
        "- run the main agent (Executive Producer)\n"
        "  python main.py visualize <main_thread_id>    "
        "- print the subagent tree of a run"
    )


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else None

    if command == "run" and len(sys.argv) > 2:
        asyncio.run(run_main_agent(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None))
    elif command == "visualize" and len(sys.argv) > 2:
        asyncio.run(visualize_subagents(sys.argv[2]))
    else:
        main()
