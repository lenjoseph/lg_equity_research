from pydantic import BaseModel


class EquityResearchRequest(BaseModel):
    """Request model for equity research endpoint."""
    
    ticker: str

