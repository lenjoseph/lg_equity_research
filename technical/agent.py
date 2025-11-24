import dotenv
from langchain_openai import ChatOpenAI

from models.api import TradeDuration
from models.trade_duration_utils import trade_duration_to_label
from shared.agent_utils import run_agent_with_tools
from shared.llm_models import LLM_MODELS
from technical.prompt import technical_research_prompt
from technical.tools import get_technical_analysis_tool

dotenv.load_dotenv()


def get_technical_sentiment(ticker: str, trade_duration: TradeDuration):
    trade_duration_label = trade_duration_to_label(trade_duration)
    prompt = f"{technical_research_prompt}\n\nAnalyze the technical indicators for ticker: {ticker}\nTrade Duration: {trade_duration_label}\n\nWhen calling the technical analysis tool, use trade_duration='{trade_duration.value}'"
    tools = [get_technical_analysis_tool]
    model = LLM_MODELS["open_ai_smart"]
    llm = ChatOpenAI(model=model)
    result = run_agent_with_tools(llm, prompt, tools)
    return result
