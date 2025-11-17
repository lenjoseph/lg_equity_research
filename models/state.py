from pydantic import BaseModel


class EquityResearchState(BaseModel):
    """State model for the equity research workflow."""
    
    ticker: str
    fundamental_sentiment: str
    technical_sentiment: str
    macro_sentiment: str
    industry_sentiment: str
    headline_sentiment: str
    combined_sentiment: str

