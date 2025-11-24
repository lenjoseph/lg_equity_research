from pydantic import BaseModel
from enum import Enum


class EquityResearchRequest(BaseModel):
    """Request model for equity research endpoint."""

    ticker: str
