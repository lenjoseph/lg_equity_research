from langchain_core.tools import Tool
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from models.tools import FundamentalsData, FundamentalsInput


def _convert_to_json_serializable(df: pd.DataFrame) -> Dict[str, Any]:
    """Convert a pandas DataFrame to a JSON-serializable dictionary."""
    if df is None:
        return None

    try:
        if df.empty:
            return None
    except (AttributeError, ValueError):
        return None

    try:
        # Convert DataFrame to dict with string keys
        df_copy = df.copy()

        # Replace inf with NaN first, then we'll handle NaN in the dict cleaning
        df_copy = df_copy.replace([np.inf, -np.inf], np.nan)

        # Convert column names (if they are Timestamps) to strings
        if isinstance(df_copy.columns, pd.DatetimeIndex):
            df_copy.columns = df_copy.columns.strftime("%Y-%m-%d")

        # Convert index (if it's a DatetimeIndex) to strings
        if isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy.index = df_copy.index.strftime("%Y-%m-%d")

        # Convert to dictionary
        result = df_copy.to_dict()

        # Recursively convert any remaining non-serializable values
        return _clean_dict_for_json(result)
    except Exception as e:
        # If conversion fails, return None rather than crashing
        print(f"Warning: Failed to convert DataFrame to JSON: {e}")
        return None


def _clean_dict_for_json(obj: Any) -> Any:
    """Recursively clean dictionary to ensure JSON serializability."""
    if isinstance(obj, dict):
        return {str(k): _clean_dict_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_clean_dict_for_json(item) for item in obj]
    elif isinstance(obj, (float, int)):
        # Check for NaN or inf - use try/except to avoid floating point exceptions
        try:
            if pd.isna(obj):
                return None
            if isinstance(obj, float) and (obj == float("inf") or obj == float("-inf")):
                return None
            return obj
        except (ValueError, TypeError, FloatingPointError):
            return None
    elif obj is None:
        return None
    else:
        # For any other type, try to check if it's NA
        try:
            if pd.isna(obj):
                return None
        except (ValueError, TypeError):
            pass
        return obj


def get_earnings_and_financial_health(ticker: str) -> FundamentalsData:
    """
    Get comprehensive earnings and financial health data for a given ticker.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')

    Returns:
        FundamentalsData: Structured output with earnings, financial statements,
        ratios, and valuation metrics for fundamental analysis.
    """
    try:

        # Create ticker object
        stock = yf.Ticker(ticker)

        # Get financial statements (run blocking calls in thread pool)
        balance_sheet_annual = stock.balance_sheet
        balance_sheet_quarterly = stock.quarterly_balance_sheet

        income_statement_annual = stock.income_stmt
        income_statement_quarterly = stock.quarterly_income_stmt
        cash_flow_annual = stock.cashflow
        cash_flow_quarterly = stock.quarterly_cashflow

        # Extract earnings data (Net Income) from income statements
        # Note: stock.earnings is deprecated, so we extract from income_stmt instead
        earnings_annual = None
        earnings_quarterly = None
        try:
            if (
                income_statement_annual is not None
                and "Net Income" in income_statement_annual.index
            ):
                earnings_annual = income_statement_annual.loc[["Net Income"]]
        except (KeyError, AttributeError):
            pass

        try:
            if (
                income_statement_quarterly is not None
                and "Net Income" in income_statement_quarterly.index
            ):
                earnings_quarterly = income_statement_quarterly.loc[["Net Income"]]
        except (KeyError, AttributeError):
            pass

        # Get company info for ratios and metrics
        info = stock.info

        # Calculate/extract key financial ratios
        ratios = {
            "P/E_ratio": info.get("trailingPE"),
            "forward_P/E": info.get("forwardPE"),
            "PEG_ratio": info.get("pegRatio"),
            "price_to_book": info.get("priceToBook"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "ROE": info.get("returnOnEquity"),
            "ROA": info.get("returnOnAssets"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "gross_margin": info.get("grossMargins"),
        }
        ratios = _clean_dict_for_json(ratios)

        # Get valuation metrics for DCF and comparables
        valuation_metrics = {
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "enterprise_to_revenue": info.get("enterpriseToRevenue"),
            "enterprise_to_ebitda": info.get("enterpriseToEbitda"),
            "trailing_eps": info.get("trailingEps"),
            "forward_eps": info.get("forwardEps"),
            "book_value": info.get("bookValue"),
            "price_to_book": info.get("priceToBook"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "beta": info.get("beta"),
            "total_revenue": info.get("totalRevenue"),
            "revenue_per_share": info.get("revenuePerShare"),
            "total_debt": info.get("totalDebt"),
            "total_cash": info.get("totalCash"),
            "free_cash_flow": info.get("freeCashflow"),
            "operating_cash_flow": info.get("operatingCashflow"),
            "ebitda": info.get("ebitda"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "dividend_yield": info.get("dividendYield"),
            "payout_ratio": info.get("payoutRatio"),
        }
        valuation_metrics = _clean_dict_for_json(valuation_metrics)

        # Compile all data
        return FundamentalsData(
            ticker=ticker,
            company_name=info.get("longName"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            earnings={
                "annual": _convert_to_json_serializable(earnings_annual),
                "quarterly": _convert_to_json_serializable(earnings_quarterly),
            },
            balance_sheet={
                "annual": _convert_to_json_serializable(balance_sheet_annual),
                "quarterly": _convert_to_json_serializable(balance_sheet_quarterly),
            },
            income_statement={
                "annual": _convert_to_json_serializable(income_statement_annual),
                "quarterly": _convert_to_json_serializable(income_statement_quarterly),
            },
            cash_flow={
                "annual": _convert_to_json_serializable(cash_flow_annual),
                "quarterly": _convert_to_json_serializable(cash_flow_quarterly),
            },
            ratios=ratios,
            valuation_metrics=valuation_metrics,
            current_price=_clean_dict_for_json(info.get("currentPrice")),
            target_price=_clean_dict_for_json(
                {
                    "mean": info.get("targetMeanPrice"),
                    "high": info.get("targetHighPrice"),
                    "low": info.get("targetLowPrice"),
                }
            ),
        )

    except Exception as e:

        import traceback

        traceback.print_exc()
        return FundamentalsData(
            ticker=ticker,
            error=str(e),
            message=f"Failed to retrieve data for {ticker}",
        )


get_fundamentals_tool = Tool(
    name="get_fundamentals_tool",
    description="Use this tool to get comprehensive fundamental analysis data for a stock. Returns earnings, financial statements (balance sheet, income statement, cash flow), key financial ratios (P/E, ROE, margins, debt ratios), and valuation metrics (market cap, EV, beta, FCF) useful for DCF and comparable company analysis.",
    func=get_earnings_and_financial_health,
    args_schema=FundamentalsInput,
)
