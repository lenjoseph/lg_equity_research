import time
from typing import Dict, Tuple, Any, Optional

import dotenv

from agents.evaluation.prompt import sentiment_evaluator_prompt
from agents.shared.llm_models import LLM_MODELS, get_openai_llm
from agents.shared.agent_utils import invoke_llm_with_metrics
from agents.shared.token_config import DEFAULT_TOKEN_CONFIG, AgentTokenConfig
from models.agent import AggregatorFeedback
from models.metrics import AgentMetrics

dotenv.load_dotenv()

AGENT_NAME = "evaluation"


def evaluate_aggregated_sentement(
    sentiment: str,
    iteration: int = 1,
    token_config: Optional[AgentTokenConfig] = None,
) -> Tuple[Dict[str, Any], AgentMetrics]:
    """
    Evaluate aggregated sentiment for compliance.

    Args:
        sentiment: The aggregated sentiment to evaluate
        iteration: The iteration number (1-based) for the evaluation loop
        token_config: Optional token configuration for this agent

    Returns:
        Tuple of (evaluation dict with compliant and feedback, AgentMetrics)
    """
    start_time = time.perf_counter()
    config = token_config or DEFAULT_TOKEN_CONFIG.evaluation
    model = LLM_MODELS["open_ai_smart"]

    prompt = f"Evaluate this sentiment for criteria compliance: {sentiment}"

    prompt += (
        f"Use these criteria as the evaluation target: {sentiment_evaluator_prompt}"
    )

    # Get cached base LLM, then wrap with structured output
    base_llm = get_openai_llm(
        model=model,
        temperature=0.0,
        max_tokens=config.max_output_tokens,
    )
    result, token_usage = invoke_llm_with_metrics(
        base_llm, prompt, AggregatorFeedback, token_budget=config.token_budget
    )

    # Check if budget was exceeded
    budget_exceeded = False
    if config.token_budget and token_usage.total_tokens > config.token_budget:
        budget_exceeded = True

    latency_ms = (time.perf_counter() - start_time) * 1000
    # Include iteration in agent name for tracking multiple loops
    agent_name = f"{AGENT_NAME}_{iteration}" if iteration > 1 else AGENT_NAME
    metrics = AgentMetrics(
        agent_name=agent_name,
        latency_ms=latency_ms,
        token_usage=token_usage,
        model=model,
        budget_exceeded=budget_exceeded,
    )

    if result is None:
        return {
            "compliant": True,
            "feedback": "Evaluation unavailable due to API error.",
        }, metrics

    return {"compliant": result.compliant, "feedback": result.feedback}, metrics
