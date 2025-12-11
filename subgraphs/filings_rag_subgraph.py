from langgraph.graph import END, StateGraph, START
from agents.filings.agent import get_filings_sentiment
from data.util.ingest_sec_filings import ensure_filings_ingested
from models.state import EquityResearchState
from models.metrics import RequestMetrics
from util.cache import create_cache_policy
from util.formating import format_sentiment_output
from util.logger import get_logger

logger = get_logger(__name__)


def filings_rag_ingestion(state: EquityResearchState) -> dict:
    """Ingest SEC filings into vector store before research agents run"""
    logger.info(f"Starting SEC filings ingestion for {state.ticker}")
    try:
        was_ingested = ensure_filings_ingested(ticker=state.ticker)
        if was_ingested:
            logger.info(f"SEC filings ingested for {state.ticker}")
        else:
            logger.info(f"SEC filings already available for {state.ticker}")
        return {"filings_ingested": True}
    except Exception as e:
        logger.error(
            f"SEC filings ingestion failed for {state.ticker}: {e}", exc_info=True
        )
        return {"filings_ingested": False}


def filings_rag_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate SEC filings research sentiment"""
    logger.info(f"Starting filings research for {state.ticker}")
    try:
        filings_sentiment, agent_metrics = get_filings_sentiment(ticker=state.ticker)
        metrics = RequestMetrics()
        metrics.add_agent_metrics(agent_metrics)
        if filings_sentiment:
            logger.info(f"Completed filings research for {state.ticker}")
            return {
                "filings_sentiment": format_sentiment_output(filings_sentiment),
                "metrics": metrics,
            }
        else:
            return {
                "filings_sentiment": "No SEC filings available for analysis.",
                "metrics": metrics,
            }
    except Exception as e:
        logger.error(f"Filings research failed for {state.ticker}: {e}", exc_info=True)
        return {
            "filings_sentiment": "Analysis unavailable due to data retrieval error."
        }


# build filings subgraph
filings_rag_builder = StateGraph(EquityResearchState)
filings_rag_builder.add_node("filings_ingestion", filings_rag_ingestion)
filings_rag_builder.add_node(
    "filings_research_agent",
    filings_rag_research_agent,
    cache_policy=create_cache_policy(ttl=86400),
)
filings_rag_builder.add_edge(START, "filings_ingestion")
filings_rag_builder.add_edge("filings_ingestion", "filings_research_agent")
filings_rag_builder.add_edge("filings_research_agent", END)
filings_rag_subgraph = filings_rag_builder.compile()
