# lg_equity_research

Agentic equity research implemented on LangGraph

# Running Locally

To run the program locally, follow these steps:

1. Initialize a python environment
2. Install deps with pip or other python package manager
3. Hydrate API keys in a new .env file (keys are specified in the env.dev file)
4. Run `python main.py` from the root of the project
5. From a separate terminal, execute the following curl command:
   curl -X POST "http://localhost:8000/research-equity" -H "Content-Type: application/json" -d '{"ticker": "TICKER"}'

# Architecture

This app implements an agentic ai architecture to compile equity research on a single stock across different analytical lenses
The app exposes a POST endpoint that receives a single stock ticker which serves as the focal point of the research.

![Architecture Diagram](/architecture.png)

The app structures ai agents as research domain specialists that perform data gathering and analysis scoped to their respective domain.
The app manages state through langgraph's graph state model.
Research agents are executed in parallel at the start of the workflow.
When all agents have executed, a aggregator agent synthesizes overall sentiment for the stock.
The graph implements a node-based cache that configures cache keys and ttl at the agent level.

# Architecture Components

- HTTP API: FastAPI
- Agentic Architecture: Langgraph
- Agent node caching: Langgraph InMemory Cache
- Agent Observability: LangSmith
- Data Typing: Pydantic
- LLM Models: OpenAI
- In-Memory Vector Store: ChromaDB
- Vector Embedding Model: HuggingFace
- Financial Data API: YFinance
- Economic Data API: Pandas Datareader

# AI Agent Descriptions

1. Technical Researcher - Focuses on technical price data over fixed timeframes
2. Fundamentals Researcher - Focuses on business fundamentals via most recent earnings results
3. Macro Economic Researcher - Focuses on the macro economic components provided by federal reserve data
4. Industry Researcher - Focuses on forecasted headwinds / tailwinds relative to the industry of the stock
5. Headline Researcher - Focuses on recent (within the last month) headlines about the stock
6. Sentiment Aggregator - Compiles overall stock sentiment based on aggregate findings of research specialist agents
