from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


LLM_MODELS = {
    "open_ai_fast": "gpt-4o-mini",
    "open_ai_smart": "gpt-5.1",
    "google_fast": "gemini-2.5-flash-lite",
    "google_smart": "gemini-2.5-flash",
}

# Default timeout for LLM API calls (in seconds)
DEFAULT_LLM_TIMEOUT = 60


@lru_cache(maxsize=8)
def get_openai_llm(
    model: str,
    temperature: float = 0.0,
    timeout: int = DEFAULT_LLM_TIMEOUT,
) -> ChatOpenAI:
    """
    Get a cached ChatOpenAI instance.

    Args:
        model: The OpenAI model name
        temperature: The temperature setting (default 0.0)
        timeout: Request timeout in seconds (default 60)

    Returns:
        Cached ChatOpenAI instance
    """
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        request_timeout=timeout,
    )


@lru_cache(maxsize=4)
def get_google_llm(
    model: str,
    temperature: float = 0.0,
    with_search_grounding: bool = False,
    timeout: int = DEFAULT_LLM_TIMEOUT,
) -> ChatGoogleGenerativeAI:
    """
    Get a cached ChatGoogleGenerativeAI instance.

    Args:
        model: The Google model name
        temperature: The temperature setting (default 0.0)
        with_search_grounding: Whether to enable Google Search grounding
        timeout: Request timeout in seconds (default 60)

    Returns:
        Cached ChatGoogleGenerativeAI instance
    """
    model_kwargs = {}
    if with_search_grounding:
        model_kwargs["tools"] = [{"google_search_retrieval": {}}]

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        timeout=timeout,
        model_kwargs=model_kwargs if model_kwargs else None,
    )
