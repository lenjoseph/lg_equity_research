import time
from typing import Tuple

import dotenv

from agents.shared.agent_utils import run_agent_with_tools
from agents.shared.llm_models import LLM_MODELS, get_openai_llm
from agents.technical.prompt import technical_research_prompt
from agents.technical.tools import get_technical_analysis_tool
from models.agent import TechnicalSentimentOutput
from models.metrics import AgentMetrics

dotenv.load_dotenv()

AGENT_NAME = "technical"


def get_technical_sentiment(
    ticker: str,
) -> Tuple[TechnicalSentimentOutput, AgentMetrics]:
    """
    Generate technical sentiment analysis for a ticker.

    Returns:
        Tuple of (TechnicalSentimentOutput, AgentMetrics)
    """
    start_time = time.perf_counter()
    model = LLM_MODELS["open_ai_smart"]

    prompt = f"{technical_research_prompt}\n\nAnalyze the technical indicators for ticker: {ticker}"
    tools = [get_technical_analysis_tool]
    llm = get_openai_llm(model=model, temperature=0.0)
    result, token_usage = run_agent_with_tools(
        llm, prompt, tools, TechnicalSentimentOutput, track_tokens=True
    )

    latency_ms = (time.perf_counter() - start_time) * 1000
    metrics = AgentMetrics(
        agent_name=AGENT_NAME,
        latency_ms=latency_ms,
        token_usage=token_usage,
        model=model,
    )
    return result, metrics
