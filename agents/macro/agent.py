import time
from typing import Optional, Tuple

import dotenv

from agents.macro.prompt import macro_research_prompt
from agents.macro.tools import get_macro_data_tool
from agents.shared.agent_utils import run_agent_with_tools
from agents.shared.llm_models import LLM_MODELS, get_openai_llm
from agents.shared.token_config import DEFAULT_TOKEN_CONFIG, AgentTokenConfig
from models.agent import MacroSentimentOutput
from models.metrics import AgentMetrics

dotenv.load_dotenv()

AGENT_NAME = "macro"


def get_macro_sentiment(
    token_config: Optional[AgentTokenConfig] = None,
) -> Tuple[MacroSentimentOutput, AgentMetrics]:
    """
    Generate macro sentiment analysis.

    Args:
        token_config: Optional token configuration for this agent

    Returns:
        Tuple of (MacroSentimentOutput, AgentMetrics)
    """
    start_time = time.perf_counter()
    config = token_config or DEFAULT_TOKEN_CONFIG.macro
    model = LLM_MODELS["open_ai_smart"]

    prompt = macro_research_prompt
    tools = [get_macro_data_tool]
    llm = get_openai_llm(
        model=model,
        temperature=0.0,
        max_tokens=config.max_output_tokens,
    )
    result, token_usage = run_agent_with_tools(
        llm, prompt, tools, MacroSentimentOutput,
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
