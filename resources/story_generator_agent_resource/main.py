from llm_config.main import llm_config
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage
from logging_config import log_message, log_important_step


agent_config = llm_config["story_generator_agent"]
llm = init_chat_model(agent_config["MODEL"], temperature=agent_config["TEMPERATURE"])


async def story_generator(question: str) -> str:

    log_important_step("Story generator agent invoked", f"Question: {question[:80]}")

    llm_config_extra = agent_config.get("CONFIG", {})

    response = await llm.ainvoke(
        [
            SystemMessage(content=agent_config["SYSTEM_PROMPT"]),
            HumanMessage(content=question),
        ],
        **llm_config_extra,
    )

    final_content = response.content

    log_message("Story generator agent finished")

    return final_content
