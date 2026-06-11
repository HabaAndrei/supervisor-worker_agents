# llm_config

Central configuration for every LLM agent in the project. Each agent has its own YAML file in this folder, and all files are loaded automatically into a single `llm_config` dictionary at import time.

## How it works

[main.py](main.py) scans this directory for `.yml` / `.yaml` files and loads them with `yaml.safe_load`. Each file must contain a top-level `NAME` property — that value becomes the key in the resulting dictionary:

```python
from llm_config.main import llm_config

agent_config = llm_config["screenwriter_agent"]
```

Files that are empty, fail to parse, or are missing `NAME` are skipped and the problem is logged via `logging_config.log_error` — they do not crash the application.

## Configuration files

| File | Agent |
|------|-------|
| `executive_producer_agent.yml` | Executive Producer (supervisor of the whole pipeline) |
| `screenwriter_agent.yml` | Screenwriter (narrative foundation) |
| `story_generator_agent.yml` | Story Generator (plot, genre, setting, conflict) |
| `character_creator_agent.yml` | Character Creator |
| `director_agent.yml` | Director |
| `scene_planner_agent.yml` | Scene Planner |

## YAML schema

Every config file follows the same structure:

```yaml
NAME: screenwriter_agent          # Required. Key used to look up the config in llm_config.

SYSTEM_PROMPT: |                  # System prompt injected on every LLM call for this agent.
    You are Screenwriter.
    ...

MODEL: openai:gpt-5.4-mini        # Model id in "provider:model" format,
                                  # passed to langchain's init_chat_model.

TEMPERATURE: 0                    # Sampling temperature for the model.

CONFIG:                           # Extra provider options.
    prompt_cache_key: key_screenwriter_agent
```

| Key | Required | Used by | Description |
|-----|----------|---------|-------------|
| `NAME` | Yes | `main.py` loader | Dictionary key for the agent's config. Must be unique across files. |
| `SYSTEM_PROMPT` | Yes | Agent `llm_call` nodes | Prepended as a `SystemMessage` to the conversation on every LLM invocation. |
| `MODEL` | Yes | Agent module setup | `provider:model` string consumed by `init_chat_model`. |
| `TEMPERATURE` | Yes | Agent module setup | Sampling temperature. |
| `CONFIG` | No | Agent LLM calls | Additional provider options passed as keyword arguments on every LLM invocation (e.g. `prompt_cache_key` for OpenAI prompt caching). |

## Adding a new agent

1. Create `<agent_name>.yml` in this folder following the schema above.
2. Set a unique `NAME` — this is how the agent code will look it up.
3. Read it from your agent module:

```python
from llm_config.main import llm_config

agent_config = llm_config["<agent_name>"]

llm = init_chat_model(agent_config["MODEL"], temperature=agent_config["TEMPERATURE"])
```

No registration step is needed — any valid YAML file in this directory is picked up automatically.
