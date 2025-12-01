from pydantic import BaseModel
from enum import Enum

from models.state import TradeDuration


class EquityResearchRequest(BaseModel):
    """Request model for equity research endpoint."""

    ticker: str
    trade_duration: TradeDuration
