from pydantic import BaseModel


class EquityResearchRequest(BaseModel):
    """Request model for equity research endpoint."""

    ticker: str
    trade_duration_days: int
