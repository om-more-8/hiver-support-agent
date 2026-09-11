"""
Thin wrapper so the rest of the codebase doesn't care whether you're
using Groq or OpenRouter — both expose OpenAI-compatible endpoints.
"""
from openai import OpenAI
from src.config import LLM_PROVIDER, LLM_MODEL, GROQ_API_KEY, OPENROUTER_API_KEY

_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


def get_client() -> OpenAI:
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in .env")
        return OpenAI(api_key=GROQ_API_KEY, base_url=_BASE_URLS["groq"])
    elif LLM_PROVIDER == "openrouter":
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not set in .env")
        return OpenAI(api_key=OPENROUTER_API_KEY, base_url=_BASE_URLS["openrouter"])
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER} (use 'groq' or 'openrouter')")


def chat(prompt: str, system: str = "", temperature: float = 0.0) -> str:
    client = get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()