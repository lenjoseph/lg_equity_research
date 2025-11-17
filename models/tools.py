from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any


# Fundamentals Models
class FundamentalsData(BaseModel):
    """Fundamental analysis data for a stock ticker."""

    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: Optional[str] = Field(None, description="Company full name")
    sector: Optional[str] = Field(None, description="Company sector")
    industry: Optional[str] = Field(None, description="Company industry")

    # Earnings data
    earnings: Optional[Dict[str, Any]] = Field(
        None, description="Earnings data (annual and quarterly)"
    )

    # Financial statements
    balance_sheet: Optional[Dict[str, Any]] = Field(
        None, description="Balance sheet data (annual and quarterly)"
    )
    income_statement: Optional[Dict[str, Any]] = Field(
        None, description="Income statement data (annual and quarterly)"
    )
    cash_flow: Optional[Dict[str, Any]] = Field(
        None, description="Cash flow statement (annual and quarterly)"
    )

    # Financial ratios
    ratios: Optional[Dict[str, Any]] = Field(
        None,
        description="Key financial ratios (P/E, ROE, debt-to-equity, margins, etc.)",
    )

    # Valuation metrics
    valuation_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Valuation metrics for DCF and comparable analysis (market cap, EV, beta, etc.)",
    )

    # Price data
    current_price: Optional[float] = Field(None, description="Current stock price")
    target_price: Optional[Dict[str, Any]] = Field(
        None, description="Analyst target prices (mean, high, low)"
    )

    # Error handling
    error: Optional[str] = Field(None, description="Error message if analysis failed")
    message: Optional[str] = Field(None, description="Additional error context")


class FundamentalsInput(BaseModel):
    """Input schema for fundamentals analysis tool."""

    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL', 'MSFT')")


# Technical Analysis Models
class TechnicalAnalysis(BaseModel):
    """Technical analysis data for a stock ticker."""

    ticker: str = Field(..., description="Stock ticker")
    current_price: float = Field(..., description="Current stock price")
    analysis_date: str = Field(
        ..., description="Date and time of analysis in YYYY-MM-DD HH:MM:SS format"
    )

    # Mid-term indicators (daily data)
    mid_rsi: Optional[float] = Field(None, description="14-day Relative Strength Index")
    mid_rsi_signal: str = Field(
        ..., description="RSI signal: oversold, neutral, overbought, or unknown"
    )
    mid_sma_50: Optional[float] = Field(
        None, description="50-day Simple Moving Average"
    )
    mid_sma_trend: int = Field(
        ..., description="Price vs 50 SMA: 1 (above), -1 (below), 0 (unknown)"
    )
    mid_stoch_k: Optional[float] = Field(None, description="Stochastic %K oscillator")
    mid_stoch_signal: str = Field(
        ..., description="Stochastic signal: oversold, neutral, overbought, or unknown"
    )

    # Macro-term indicators (weekly data)
    macro_sma_200: Optional[float] = Field(
        None, description="200-period Simple Moving Average (weekly)"
    )
    macro_sma_trend: int = Field(
        ..., description="Price vs 200 SMA: 1 (above), -1 (below), 0 (unknown)"
    )
    macro_macd: Optional[float] = Field(None, description="MACD line value (weekly)")
    macro_macd_signal: int = Field(
        ..., description="MACD vs Signal: 1 (bullish), -1 (bearish), 0 (unknown)"
    )
    macro_bb_position: Optional[float] = Field(
        None, description="Position within Bollinger Bands as percentage (0-100)"
    )
    macro_bb_signal: str = Field(
        ...,
        description="Bollinger Bands signal: oversold, neutral, overbought, or unknown",
    )

    # Overall assessment
    overall_sentiment: float = Field(
        ..., description="Overall sentiment score from -1 (bearish) to 1 (bullish)"
    )

    error: Optional[str] = Field(None, description="Error message if analysis failed")


class TechnicalAnalysisInput(BaseModel):
    """Input schema for technical analysis tool."""

    ticker: str = Field(..., description="Stock ticker (e.g., 'AAPL', 'MSFT')")
    period: str = Field(
        default="2y", description="Data period for analysis (e.g., '1y', '2y', '5y')"
    )


# Macro Data Models
class HistoricalDataPoint(BaseModel):
    """A single historical data point with date and value."""

    date: str = Field(..., description="Date in YYYY-MM-DD format")
    value: float = Field(..., description="The value at this date")


class IndicatorData(BaseModel):
    """Data for a single macroeconomic indicator."""

    latest_value: float = Field(..., description="Most recent value")
    latest_date: str = Field(
        ..., description="Date of most recent value in YYYY-MM-DD format"
    )
    historical_data: list[HistoricalDataPoint] = Field(
        ..., description="Historical time series data"
    )
    quarterly_change_pct_points: Optional[float] = Field(
        None,
        description="Quarterly change in percentage points (for rate-based indicators)",
    )
    monthly_change_percent: Optional[float] = Field(
        None, description="Monthly percentage change (for index-based indicators)"
    )
    yoy_inflation_rate: Optional[float] = Field(
        None, description="Year-over-year inflation rate (CPI only)"
    )
    error: Optional[str] = Field(
        None, description="Error message if data retrieval failed"
    )


class MacroDataResponse(BaseModel):
    """Response containing macroeconomic data from FRED."""

    timestamp: str = Field(..., description="Timestamp when data was retrieved")
    data: dict[str, IndicatorData] = Field(
        ..., description="Dictionary of indicator names to their data"
    )
    error: Optional[str] = Field(
        None, description="Error message if overall retrieval failed"
    )


class MacroDataInput(BaseModel):
    """Input schema for the macro data tool. No parameters required."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
