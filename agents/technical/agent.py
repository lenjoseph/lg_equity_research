import time
from typing import Optional, Tuple

import dotenv

from agents.shared.agent_utils import run_agent_with_tools
from agents.shared.llm_models import LLM_MODELS, get_openai_llm
from agents.shared.token_config import DEFAULT_TOKEN_CONFIG, AgentTokenConfig
from agents.technical.prompt import technical_research_prompt
from agents.technical.tools import get_technical_analysis_tool
from models.agent import TechnicalSentimentOutput
from models.metrics import AgentMetrics

dotenv.load_dotenv()

AGENT_NAME = "technical"


def get_technical_sentiment(
    ticker: str,
    token_config: Optional[AgentTokenConfig] = None,
) -> Tuple[TechnicalSentimentOutput, AgentMetrics]:
    """
    Generate technical sentiment analysis for a ticker.

    Args:
        ticker: Stock ticker symbol
        token_config: Optional token configuration for this agent

    Returns:
        Tuple of (TechnicalSentimentOutput, AgentMetrics)
    """
    start_time = time.perf_counter()
    config = token_config or DEFAULT_TOKEN_CONFIG.technical
    model = LLM_MODELS["open_ai_smart"]

    prompt = f"{technical_research_prompt}\n\nAnalyze the technical indicators for ticker: {ticker}"
    tools = [get_technical_analysis_tool]
    llm = get_openai_llm(
        model=model,
        temperature=0.0,
        max_tokens=config.max_output_tokens,
    )
    result, token_usage = run_agent_with_tools(
        llm, prompt, tools, TechnicalSentimentOutput,
        track_tokens=True, token_budget=config.token_budget
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
