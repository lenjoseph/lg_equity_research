import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from agents.macro.tools import (
    _fetch_indicator_data,
    _calculate_yoy_inflation,
    get_macro_data,
    INDICATORS_CONFIG,
)
from models.tools import IndicatorData, HistoricalDataPoint


class TestFetchIndicatorData:
    @pytest.fixture
    def mock_data(self):
        dates = pd.date_range(start="2023-01-01", periods=3, freq="M")
        return pd.DataFrame({"value": [100.0, 102.0, 105.0]}, index=dates)

    @patch("agents.macro.tools.pdr.DataReader")
    def test_fetch_indicator_data_rate_type(self, mock_pdr, mock_data):
        mock_pdr.return_value = mock_data

        result = _fetch_indicator_data(
            code="TEST",
            is_rate=True,  # Like GDP
            change_label="quarterly_change_pct_points",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 31),
        )

        assert result.latest_value == 105.0
        assert result.quarterly_change_pct_points == 3.0  # 105.0 - 102.0
        assert len(result.historical_data) == 3
        assert result.error is None
        mock_pdr.assert_called_once()

    @patch("agents.macro.tools.pdr.DataReader")
    def test_fetch_indicator_data_percentage_type(self, mock_pdr, mock_data):
        mock_pdr.return_value = mock_data

        result = _fetch_indicator_data(
            code="TEST",
            is_rate=False,  # Like CPI
            change_label="monthly_change_percent",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 31),
        )

        # Change should be ((105 - 102) / 102) * 100 = 2.9411...
        assert result.latest_value == 105.0
        assert result.monthly_change_percent == 2.94  # Rounded to 2 decimals
        assert len(result.historical_data) == 3

    @patch("agents.macro.tools.pdr.DataReader")
    def test_fetch_indicator_no_data(self, mock_pdr):
        mock_pdr.return_value = pd.DataFrame()

        result = _fetch_indicator_data(
            code="EMPTY",
            is_rate=False,
            change_label="change",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 1),
        )

        assert result.latest_value == 0.0
        assert result.error == "No data available"
        assert len(result.historical_data) == 0

    @patch("agents.macro.tools.pdr.DataReader")
    def test_fetch_indicator_error(self, mock_pdr):
        mock_pdr.side_effect = Exception("FRED Error")

        result = _fetch_indicator_data(
            code="ERROR",
            is_rate=False,
            change_label="change",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 1),
        )

        assert result.error == "FRED Error"
        assert result.latest_value == 0.0


class TestCalculateYoyInflation:
    @patch("agents.macro.tools.pdr.DataReader")
    def test_calculate_yoy_inflation_success(self, mock_pdr):
        # Setup mock for 1 year ago data
        mock_pdr.return_value = pd.DataFrame(
            {"value": [100.0]}, index=[datetime.now() - timedelta(days=365)]
        )

        current_cpi = 105.0
        inflation = _calculate_yoy_inflation(current_cpi, datetime.now())

        # ((105 - 100) / 100) * 100 = 5.0%
        assert inflation == 5.0

    @patch("agents.macro.tools.pdr.DataReader")
    def test_calculate_yoy_inflation_no_data(self, mock_pdr):
        mock_pdr.return_value = pd.DataFrame()

        result = _calculate_yoy_inflation(105.0, datetime.now())
        assert result is None

    @patch("agents.macro.tools.pdr.DataReader")
    def test_calculate_yoy_inflation_error(self, mock_pdr):
        mock_pdr.side_effect = Exception("API Error")

        result = _calculate_yoy_inflation(105.0, datetime.now())
        assert result is None


class TestGetMacroData:
    @patch("agents.macro.tools._fetch_indicator_data")
    @patch("agents.macro.tools._calculate_yoy_inflation")
    def test_get_macro_data_success(self, mock_calc_yoy, mock_fetch):
        # Setup real IndicatorData return
        success_result = IndicatorData(
            latest_value=100.0, latest_date="2023-01-01", historical_data=[], error=None
        )

        mock_fetch.return_value = success_result
        mock_calc_yoy.return_value = 3.5

        result = get_macro_data()

        assert result.error is None
        assert "gdp_growth" in result.data
        assert "inflation_cpi" in result.data
        assert "consumer_sentiment" in result.data

        # Verify calls
        assert mock_fetch.call_count == len(INDICATORS_CONFIG)
        # Verify inflation calc was called
        mock_calc_yoy.assert_called_once()
        # Verify inflation was added (CPI is one of the indicators)
        assert result.data["inflation_cpi"].yoy_inflation_rate == 3.5

    @patch("agents.macro.tools._fetch_indicator_data")
    def test_get_macro_data_partial_failure(self, mock_fetch):
        # One success, one failure
        success_result = IndicatorData(
            latest_value=100.0, latest_date="2023-01-01", historical_data=[], error=None
        )
        failure_result = IndicatorData(
            latest_value=0.0, latest_date="", historical_data=[], error="Failed"
        )

        mock_fetch.side_effect = [success_result, failure_result, success_result]

        result = get_macro_data()

        assert result.error is None
        assert len(result.data) == 3
        # The order of keys in INDICATORS_CONFIG is gdp, cpi, sentiment
        # side_effect maps to these calls
        # 1. gdp -> success
        # 2. cpi -> failure
        # 3. sentiment -> success

        # Should still contain the failed result structure
        assert result.data["inflation_cpi"].error == "Failed"

    @patch("agents.macro.tools._fetch_indicator_data")
    def test_get_macro_data_global_exception(self, mock_fetch):
        mock_fetch.side_effect = Exception("Global Crash")

        result = get_macro_data()

        assert "Failed to retrieve macro data" in result.error
        assert result.data == {}
