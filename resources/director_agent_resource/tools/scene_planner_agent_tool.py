from ...scene_planner_agent_resource.main import scene_planner
from ...utils.state_declaration import GeneralChatAgentState
from langchain.tools import tool


@tool(
    description=(
        "Delegate scene planning to the Scene Planner agent. It converts an "
        "approved story into a scene-by-scene production plan: Scene List, "
        "Scene Descriptions, and Production Notes. The Scene Planner has no "
        "memory and only knows what you pass it, so 'task' must contain the "
        "FULL approved story, including characters and any constraints."
    )
)
async def call_scene_planner_agent(
    task: str,
    state: GeneralChatAgentState = None,
):
    return await scene_planner(task)
