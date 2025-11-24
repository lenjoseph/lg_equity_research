from langchain_core.tools import Tool
import yfinance as yf
import pandas as pd
from datetime import datetime

from models.tools import TechnicalAnalysis, TechnicalAnalysisInput


def calculate_sma(data: pd.Series, length: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return data.rolling(window=length).mean()


def calculate_rsi(data: pd.Series, length: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    return pd.DataFrame({"K": k, "D": d})


def calculate_macd(
    data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd - macd_signal
    return pd.DataFrame(
        {"MACD": macd, "Signal": macd_signal, "Histogram": macd_histogram}
    )


def calculate_bollinger_bands(
    data: pd.Series, length: int = 20, std_dev: float = 2.0
) -> pd.DataFrame:
    """Calculate Bollinger Bands"""
    middle = data.rolling(window=length).mean()
    std = data.rolling(window=length).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return pd.DataFrame({"Upper": upper, "Middle": middle, "Lower": lower})


def safe_float(value, default=None):
    """Convert to float, return default if NaN"""
    if pd.isna(value):
        return default
    return float(value)


def get_signal(value, low_threshold, high_threshold, default="unknown"):
    """Get signal, return default if NaN"""
    if pd.isna(value):
        return default
    if value < low_threshold:
        return "oversold"
    elif value > high_threshold:
        return "overbought"
    else:
        return "neutral"


def safe_compare(val1, val2, default=0):
    """Compare two values, return default if either is NaN"""
    if pd.isna(val1) or pd.isna(val2):
        return default
    return 1 if val1 > val2 else -1


def get_bollinger_signal(price, bb_upper, bb_lower):
    """Determine Bollinger Bands signal"""
    if pd.isna(bb_lower) or pd.isna(bb_upper):
        return "unknown"
    if price < bb_lower:
        return "oversold"
    elif price > bb_upper:
        return "overbought"
    else:
        return "neutral"


def get_technical_analysis(ticker: str) -> TechnicalAnalysis:
    """
    Performs technical analysis on a stock with standard indicators.

    Args:
        ticker (str): Stock ticker (e.g., 'AAPL').

    Returns:
        TechnicalAnalysis: Structured output with indicators, signals, and sentiment.
    """
    try:
        # Standard indicator periods
        periods = {
            "short_sma": 20,
            "long_sma": 50,
            "rsi": 14,
            "stochastic": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb": 20,
            "interval": "1d",
            "history_period": "1y",
        }

        stock = yf.Ticker(ticker)
        data = stock.history(
            period=periods["history_period"], interval=periods["interval"]
        )

        if data.empty:
            raise ValueError(f"No data found for {ticker}")

        # Add technical indicators with calibrated periods
        add_technical_indicators(data, periods)

        # Current values (latest close)
        latest = data.iloc[-1]
        current_price = latest["Close"]

        # Calculate trends and signals
        short_sma_trend = safe_compare(current_price, latest["Short_SMA"])
        long_sma_trend = safe_compare(current_price, latest["Long_SMA"])
        macd_signal = safe_compare(latest["MACD"], latest["MACD_Signal"])

        # Overall sentiment score (-1 to 1)
        # Filter out 0 values that indicate missing/unknown data
        signals = [short_sma_trend, long_sma_trend, macd_signal]
        valid_signals = [s for s in signals if s != 0]
        if valid_signals:
            overall_sentiment = sum(valid_signals) / len(valid_signals)
        else:
            overall_sentiment = 0.0  # Neutral if no valid signals

        # Calculate Bollinger Band position
        bb_position = None
        if not pd.isna(latest["BB_Upper"]) and not pd.isna(latest["BB_Lower"]):
            bb_range = latest["BB_Upper"] - latest["BB_Lower"]
            if bb_range > 0:
                bb_position = safe_float(
                    (current_price - latest["BB_Lower"]) / bb_range * 100
                )

        return TechnicalAnalysis(
            ticker=ticker,
            current_price=float(current_price),
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # Short-term indicators
            short_rsi=safe_float(latest["RSI"]),
            short_rsi_signal=get_signal(latest["RSI"], 30, 70),
            short_sma=safe_float(latest["Short_SMA"]),
            short_sma_trend=short_sma_trend,
            short_stoch_k=safe_float(latest["Stoch_K"]),
            short_stoch_signal=get_signal(latest["Stoch_K"], 20, 80),
            # Long-term indicators
            long_sma=safe_float(latest["Long_SMA"]),
            long_sma_trend=long_sma_trend,
            macd=safe_float(latest["MACD"]),
            macd_signal_value=macd_signal,
            bb_position=bb_position,
            bb_signal=get_bollinger_signal(
                current_price, latest["BB_Upper"], latest["BB_Lower"]
            ),
            # Period information
            rsi_period=periods["rsi"],
            short_sma_period=periods["short_sma"],
            long_sma_period=periods["long_sma"],
            stochastic_period=periods["stochastic"],
            bb_period=periods["bb"],
            # Overall
            overall_sentiment=overall_sentiment,
        )

    except Exception as e:
        return TechnicalAnalysis(
            ticker=ticker,
            current_price=0.0,
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            short_rsi_signal="unknown",
            short_sma_trend=0,
            short_stoch_signal="unknown",
            long_sma_trend=0,
            macd_signal_value=0,
            bb_signal="unknown",
            overall_sentiment=-1.0,
            error=str(e),
        )


def add_technical_indicators(data: pd.DataFrame, periods: dict):
    """Add technical indicators to data using specified periods"""
    data["Short_SMA"] = calculate_sma(data["Close"], periods["short_sma"])
    data["Long_SMA"] = calculate_sma(data["Close"], periods["long_sma"])
    data["RSI"] = calculate_rsi(data["Close"], periods["rsi"])
    stoch = calculate_stochastic(
        data["High"], data["Low"], data["Close"], k_period=periods["stochastic"]
    )
    data["Stoch_K"] = stoch["K"]
    macd = calculate_macd(
        data["Close"],
        fast=periods["macd_fast"],
        slow=periods["macd_slow"],
        signal=periods["macd_signal"],
    )
    data["MACD"] = macd["MACD"]
    data["MACD_Signal"] = macd["Signal"]
    bb = calculate_bollinger_bands(data["Close"], periods["bb"])
    data["BB_Upper"] = bb["Upper"]
    data["BB_Lower"] = bb["Lower"]
    data["BB_Mid"] = bb["Middle"]


get_technical_analysis_tool = Tool(
    name="get_technical_analysis_tool",
    description="Use this tool to perform technical analysis on a stock. Provide the ticker. The tool calculates RSI, Moving Averages, MACD, Stochastic Oscillator, and Bollinger Bands. Returns structured data with buy/sell signals and overall sentiment.",
    func=get_technical_analysis,
    args_schema=TechnicalAnalysisInput,
)
