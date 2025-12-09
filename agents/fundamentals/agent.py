import time
from typing import Any, Dict, Optional, Tuple

import dotenv

from agents.fundamentals.prompt import fundamentals_research_prompt
from agents.fundamentals.tools import (
    get_fundamentals_tool,
    get_earnings_and_financial_health,
)
from agents.shared.agent_utils import run_agent_with_tools, invoke_llm_with_metrics
from agents.shared.llm_models import LLM_MODELS, get_openai_llm
from models.agent import FundamentalSentimentOutput
from models.metrics import AgentMetrics, TokenUsage

dotenv.load_dotenv()

AGENT_NAME = "fundamental"


def get_fundamental_sentiment(
    ticker: str, cached_info: Optional[Dict[str, Any]] = None
) -> Tuple[FundamentalSentimentOutput, AgentMetrics]:
    """
    Generate fundamental sentiment analysis for a ticker.

    Args:
        ticker: Stock ticker symbol
        cached_info: Optional pre-fetched yfinance ticker.info to avoid duplicate API calls

    Returns:
        Tuple of (FundamentalSentimentOutput, AgentMetrics)
    """
    start_time = time.perf_counter()
    model = LLM_MODELS["open_ai_smart"]
    llm = get_openai_llm(model=model, temperature=0.0)

    if cached_info is not None:
        # Use cached info - call function directly instead of via tool
        fundamentals_data = get_earnings_and_financial_health(
            ticker=ticker, cached_info=cached_info
        )
        # Inject the data directly into the prompt for analysis
        prompt = f"{fundamentals_research_prompt}\n\n"
        prompt += f"Analyze the business fundamentals for ticker: {ticker}\n\n"
        prompt += f"Here is the fundamental data:\n{fundamentals_data.model_dump_json(indent=2)}"
        result, token_usage = invoke_llm_with_metrics(
            llm, prompt, FundamentalSentimentOutput
        )
    else:
        # No cached info - use tool-calling approach
        prompt = f"{fundamentals_research_prompt}\n\nAnalyze the business fundamentals for ticker: {ticker}"
        tools = [get_fundamentals_tool]
        result, token_usage = run_agent_with_tools(
            llm, prompt, tools, FundamentalSentimentOutput, track_tokens=True
        )

    latency_ms = (time.perf_counter() - start_time) * 1000
    metrics = AgentMetrics(
        agent_name=AGENT_NAME,
        latency_ms=latency_ms,
        token_usage=token_usage,
        model=model,
    )
    return result, metrics
