import os
import yaml
from logging_config import log_error


def load_yml_configs(folder_path: str):
    """Load YAML configuration files for LLM agents."""
    configs = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".yml") or filename.endswith(".yaml"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                if not data or "NAME" not in data:
                    log_error(
                        "Invalid config file",
                        details=f"File {filename} missing 'NAME' property",
                    )
                    continue

                name = data["NAME"]
                configs[name] = data
            except Exception as e:
                log_error(f"Failed to load config file {filename}", error=e)

    return configs


llm_config = load_yml_configs(os.path.dirname(os.path.abspath(__file__)))