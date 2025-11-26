import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from agents.technical.tools import (
    calculate_sma,
    calculate_rsi,
    calculate_stochastic,
    calculate_macd,
    calculate_bollinger_bands,
    safe_float,
    get_signal,
    safe_compare,
    get_bollinger_signal,
    get_technical_analysis,
    DEFAULT_PERIODS,
)


class TestTechnicalIndicators:
    @pytest.fixture
    def price_data(self):
        # Create a simple price series for testing
        return pd.Series([10, 11, 12, 11, 10, 9, 10, 11, 12, 13, 14, 15, 14, 13, 12])

    @pytest.fixture
    def ohlc_data(self):
        # Create OHLC DataFrame for testing
        dates = pd.date_range(start="2023-01-01", periods=15)
        return pd.DataFrame(
            {
                "Open": [10] * 15,
                "High": [15] * 15,
                "Low": [5] * 15,
                "Close": [10, 11, 12, 11, 10, 9, 10, 11, 12, 13, 14, 15, 14, 13, 12],
            },
            index=dates,
        )

    def test_calculate_sma(self, price_data):
        # Test SMA with window 3
        sma = calculate_sma(price_data, 3)
        assert len(sma) == len(price_data)
        # First 2 values should be NaN
        assert pd.isna(sma[0])
        assert pd.isna(sma[1])
        # Third value should be (10+11+12)/3 = 11
        assert sma[2] == 11.0

    def test_calculate_rsi(self):
        # Create a series with clear up/down movement
        # 5 days up, 5 days down
        data = pd.Series([10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10])
        rsi = calculate_rsi(data, length=5)
        assert len(rsi) == len(data)
        # Should be defined after length
        assert not pd.isna(rsi.iloc[-1])
        # Value should be between 0 and 100
        assert 0 <= rsi.iloc[-1] <= 100

    def test_calculate_stochastic(self, ohlc_data):
        stoch = calculate_stochastic(
            ohlc_data["High"],
            ohlc_data["Low"],
            ohlc_data["Close"],
            k_period=5,
            d_period=3,
        )
        assert "K" in stoch.columns
        assert "D" in stoch.columns
        assert len(stoch) == len(ohlc_data)
        # Check ranges
        valid_k = stoch["K"].dropna()
        assert ((valid_k >= 0) & (valid_k <= 100)).all()

    def test_calculate_macd(self, price_data):
        macd = calculate_macd(price_data, fast=3, slow=5, signal=2)
        assert "MACD" in macd.columns
        assert "Signal" in macd.columns
        assert "Histogram" in macd.columns
        assert len(macd) == len(price_data)
        # Verify histogram is macd - signal
        diff = (macd["MACD"] - macd["Signal"]) - macd["Histogram"]
        assert (diff.abs() < 1e-10).all()

    def test_calculate_bollinger_bands(self, price_data):
        bb = calculate_bollinger_bands(price_data, length=5, std_dev=2.0)
        assert "Upper" in bb.columns
        assert "Middle" in bb.columns
        assert "Lower" in bb.columns

        # Verify Upper > Middle > Lower (ignoring NaNs)
        valid = bb.dropna()
        assert (valid["Upper"] >= valid["Middle"]).all()
        assert (valid["Middle"] >= valid["Lower"]).all()


class TestHelperFunctions:
    def test_safe_float(self):
        assert safe_float(10) == 10.0
        assert safe_float("10.5") == 10.5
        assert safe_float(np.nan) is None
        assert safe_float("invalid") is None
        assert safe_float(None, default=0.0) == 0.0

    def test_get_signal(self):
        assert get_signal(20, 30, 70) == "oversold"
        assert get_signal(80, 30, 70) == "overbought"
        assert get_signal(50, 30, 70) == "neutral"
        assert get_signal(pd.NA, 30, 70) == "unknown"

    def test_safe_compare(self):
        assert safe_compare(10, 5) == 1
        assert safe_compare(5, 10) == -1
        assert safe_compare(5, 5) == 0
        assert safe_compare(np.nan, 5) == 0
        assert safe_compare(5, np.nan) == 0

    def test_get_bollinger_signal(self):
        assert get_bollinger_signal(100, 110, 90) == "neutral"
        assert get_bollinger_signal(85, 110, 90) == "oversold"
        assert get_bollinger_signal(115, 110, 90) == "overbought"
        assert get_bollinger_signal(100, np.nan, 90) == "unknown"


class TestGetTechnicalAnalysis:
    @patch("agents.technical.tools.yf.Ticker")
    def test_get_technical_analysis_success(self, mock_ticker):
        # Mocking the yfinance Ticker and history
        mock_stock = MagicMock()

        # Create enough data points for indicators to calculate
        # Needs > 200 points for SMA 200
        dates = pd.date_range(start="2022-01-01", periods=250)
        data = pd.DataFrame(
            {
                "Open": [100.0] * 250,
                "High": [105.0] * 250,
                "Low": [95.0] * 250,
                "Close": [100.0] * 250,
                "Volume": [1000] * 250,
            },
            index=dates,
        )

        # Make the last price higher to trigger some signals
        data.iloc[-1, data.columns.get_loc("Close")] = 110.0

        mock_stock.history.return_value = data
        mock_stock.info = {"symbol": "TEST"}
        mock_ticker.return_value = mock_stock

        result = get_technical_analysis("TEST")

        assert result.ticker == "TEST"
        assert result.current_price == 110.0
        assert result.overall_sentiment != -1.0
        assert result.sma_50 is not None
        assert result.sma_200 is not None
        assert result.rsi is not None

        # Verify yfinance was called correctly
        mock_ticker.assert_called_with("TEST")
        mock_stock.history.assert_called_once()

    @patch("agents.technical.tools.yf.Ticker")
    def test_get_technical_analysis_no_data(self, mock_ticker):
        mock_stock = MagicMock()
        mock_stock.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_stock

        result = get_technical_analysis("EMPTY")

        assert result.ticker == "EMPTY"
        assert result.overall_sentiment == -1.0
        assert "No data found" in result.error

    @patch("agents.technical.tools.yf.Ticker")
    def test_get_technical_analysis_error(self, mock_ticker):
        mock_ticker.side_effect = Exception("API Error")

        result = get_technical_analysis("ERROR")

        assert result.ticker == "ERROR"
        assert result.overall_sentiment == -1.0
        assert "API Error" in result.error
