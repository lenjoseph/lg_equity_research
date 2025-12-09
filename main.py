import os
import re
import time

# Suppress gRPC/absl logging before importing anything that uses it
# Ensure huggingface tokenizer thread pool doesn't fork when graph executes parallel agents
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from graph import research_chain
from models.api import EquityResearchRequest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

TICKER_PATTERN = re.compile(r"^[A-Z0-9.\-]{1,10}$")


def sanitize_ticker(ticker: str) -> str:
    """Sanitize and validate stock ticker input."""
    sanitized = ticker.strip().upper()

    if not sanitized:
        raise HTTPException(status_code=400, detail="Ticker cannot be empty")

    if not TICKER_PATTERN.match(sanitized):
        raise HTTPException(
            status_code=400,
            detail="Invalid ticker format. Ticker must contain only letters, numbers, dots, or hyphens (max 10 characters)",
        )

    return sanitized


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
)


@app.get("/")
def ping():
    return {"message": "Running"}


limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post("/research-equity")
@limiter.limit("10/minute")
async def research_equity(request: Request, req: EquityResearchRequest):
    start_time = time.perf_counter()
    sanitized_ticker = sanitize_ticker(req.ticker)

    res = await research_chain.ainvoke(
        {
            "ticker": sanitized_ticker,
            "trade_duration": req.trade_duration,
            "trade_direction": req.trade_direction,
        }
    )

    total_latency_ms = (time.perf_counter() - start_time) * 1000
    res.metrics.total_latency_ms = total_latency_ms

    return {
        "ticker": res.ticker,
        "sentiment_analysis": {
            "fundamental": res.fundamental_sentiment,
            "technical": res.technical_sentiment,
            "macro": res.macro_sentiment,
            "peer": res.peer_sentiment,
            "industry": res.industry_sentiment,
            "headline": res.headline_sentiment,
            "filings": res.filings_sentiment,
        },
        "combined_sentiment": res.combined_sentiment,
        "metrics": res.metrics.to_response_dict(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
