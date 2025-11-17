import dotenv

from models.state import EquityResearchState
from prompts.aggregator_prompt import research_aggregation_prompt
from constants.llm_models import LLM_MODELS
from agents.agent_utils import run_agent_with_tools

dotenv.load_dotenv()


def get_aggregated_sentiment(state: EquityResearchState):
    prompt = f"{research_aggregation_prompt}\n\nAggregate the following equity research: {state}"
    tools = []
    model = LLM_MODELS["open_ai"]
    return run_agent_with_tools(model, prompt, tools)
