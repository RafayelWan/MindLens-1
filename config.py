import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
MODEL = os.getenv("LLM_MODEL", "gpt-4o")

MEMORY_FILE = "memory.json"
MAX_MEMORY_ROUNDS = 50

TEMPERATURE = 0.8
TOP_P = 0.9

PROMPTS_DIR = "prompts"
DEFAULT_PROMPT_FILE = os.path.join(PROMPTS_DIR, "default.txt")


def load_system_prompt(name: str | None = None) -> str:
    if name is None:
        path = DEFAULT_PROMPT_FILE
    else:
        path = os.path.join(PROMPTS_DIR, f"{name}.txt")

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    return "You are a helpful assistant."


def list_prompts() -> list[str]:
    if not os.path.isdir(PROMPTS_DIR):
        return []
    return [f.replace(".txt", "") for f in os.listdir(PROMPTS_DIR) if f.endswith(".txt")]
