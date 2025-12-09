import time
from typing import Tuple

import dotenv

from agents.macro.prompt import macro_research_prompt
from agents.macro.tools import get_macro_data_tool
from agents.shared.agent_utils import run_agent_with_tools
from agents.shared.llm_models import LLM_MODELS, get_openai_llm
from models.agent import MacroSentimentOutput
from models.metrics import AgentMetrics

dotenv.load_dotenv()

AGENT_NAME = "macro"


def get_macro_sentiment() -> Tuple[MacroSentimentOutput, AgentMetrics]:
    """
    Generate macro sentiment analysis.

    Returns:
        Tuple of (MacroSentimentOutput, AgentMetrics)
    """
    start_time = time.perf_counter()
    model = LLM_MODELS["open_ai_smart"]

    prompt = macro_research_prompt
    tools = [get_macro_data_tool]
    llm = get_openai_llm(model=model, temperature=0.0)
    result, token_usage = run_agent_with_tools(
        llm, prompt, tools, MacroSentimentOutput, track_tokens=True
    )

    latency_ms = (time.perf_counter() - start_time) * 1000
    metrics = AgentMetrics(
        agent_name=AGENT_NAME,
        latency_ms=latency_ms,
        token_usage=token_usage,
        model=model,
    )
    return result, metrics
