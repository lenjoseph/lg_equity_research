import time
from datetime import datetime, timedelta
from typing import Optional, Tuple

import dotenv

from agents.headline.prompt import headline_research_prompt
from agents.shared.llm_models import LLM_MODELS, get_google_llm
from agents.shared.agent_utils import invoke_llm_with_metrics
from agents.shared.token_config import DEFAULT_TOKEN_CONFIG, AgentTokenConfig
from models.agent import HeadlineSentimentOutput
from models.metrics import AgentMetrics


dotenv.load_dotenv()

AGENT_NAME = "headline"


def get_headline_sentiment(
    business: str,
    token_config: Optional[AgentTokenConfig] = None,
) -> Tuple[HeadlineSentimentOutput, AgentMetrics]:
    """
    Get headline sentiment using Google's built-in search grounding.
    Google Search is configured via model_kwargs as it's a native Gemini feature.

    Args:
        business: Business description
        token_config: Optional token configuration for this agent

    Returns:
        Tuple of (HeadlineSentimentOutput, AgentMetrics)
    """
    start_time = time.perf_counter()
    config = token_config or DEFAULT_TOKEN_CONFIG.headline
    model = LLM_MODELS["google_fast"]

    current_date = datetime.now().strftime("%Y-%m-%d")
    cutoff_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    prompt = headline_research_prompt.format(
        business=business,
        current_date=current_date,
        cutoff_date=cutoff_date,
    )

    llm = get_google_llm(
        model=model,
        temperature=0.0,
        with_search_grounding=True,
        max_tokens=config.max_output_tokens,
    )

    result, token_usage = invoke_llm_with_metrics(
        llm, prompt, HeadlineSentimentOutput, token_budget=config.token_budget
    )

    # Check if budget was exceeded
    budget_exceeded = False
    if config.token_budget and token_usage.total_tokens > config.token_budget:
        budget_exceeded = True

    latency_ms = (time.perf_counter() - start_time) * 1000
    metrics = AgentMetrics(
        agent_name=AGENT_NAME,
        latency_ms=latency_ms,
        token_usage=token_usage,
        model=model,
        budget_exceeded=budget_exceeded,
    )
    return result, metrics
