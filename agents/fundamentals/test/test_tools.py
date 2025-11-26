import numpy as np
import pandas as pd
import pytest
from agents.fundamentals.tools import (
    _clean_dict_for_json,
    _convert_df_to_dict,
    _process_dataframe,
    _get_earnings_data,
)


class TestCleanDictForJson:
    def test_basic_types(self):
        data = {"a": 1, "b": 1.5, "c": "string", "d": True, "e": None}
        assert _clean_dict_for_json(data) == data

    def test_nested_structures(self):
        data = {
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "mixed": [{"k": "v"}, 123],
        }
        assert _clean_dict_for_json(data) == data

    def test_numpy_types(self):
        data = {
            "float": np.float64(1.5),
            "int": np.int64(42),
            "array": [np.float64(1.1), np.int64(2)],
        }
        cleaned = _clean_dict_for_json(data)
        assert isinstance(cleaned["float"], float)
        assert isinstance(cleaned["int"], int)
        assert isinstance(cleaned["array"][0], float)
        assert isinstance(cleaned["array"][1], int)

    def test_nan_and_inf(self):
        data = {"nan": np.nan, "inf": np.inf, "neg_inf": -np.inf, "nested": [np.nan, 1]}
        cleaned = _clean_dict_for_json(data)
        assert cleaned["nan"] is None
        assert cleaned["inf"] is None
        assert cleaned["neg_inf"] is None
        assert cleaned["nested"][0] is None

    def test_pd_na(self):
        data = {"na": pd.NA}
        cleaned = _clean_dict_for_json(data)
        assert cleaned["na"] is None


class TestConvertDfToDict:
    def test_empty_none(self):
        assert _convert_df_to_dict(None) is None
        assert _convert_df_to_dict(pd.DataFrame()) is None

    def test_datetime_handling(self):
        dates = pd.date_range(start="2023-01-01", periods=2)
        df = pd.DataFrame({"col1": [1, 2]}, index=dates)
        # Add datetime columns
        df.columns = pd.date_range(start="2023-01-01", periods=1)

        result = _convert_df_to_dict(df)
        assert "2023-01-01" in result
        # Check if index dates became keys in the inner dict (implied by to_dict default)
        # to_dict() default is dict like {col: {index: value}}
        # But wait, the function converts columns and index to strings.

        # Let's verify structure. to_dict() by default is 'dict' orientation (column -> index -> value)
        # If columns are dates, they become keys.
        assert "2023-01-01" in result  # Column name
        assert "2023-01-01" in result["2023-01-01"]  # Index name
        assert result["2023-01-01"]["2023-01-01"] == 1

    def test_basic_dataframe(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["row1", "row2"])
        result = _convert_df_to_dict(df)
        assert result == {"A": {"row1": 1, "row2": 2}, "B": {"row1": 3, "row2": 4}}

    def test_clean_values_in_df(self):
        df = pd.DataFrame({"A": [np.nan, np.inf]}, index=["r1", "r2"])
        result = _convert_df_to_dict(df)
        assert result["A"]["r1"] is None
        assert result["A"]["r2"] is None


class TestProcessDataframe:
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "2023": [100, 200, 300],
                "2022": [110, 210, 310],
                "2021": [120, 220, 320],
                "2020": [130, 230, 330],
            },
            index=["Revenue", "Cost", "Profit"],
        )

    def test_none_empty(self):
        assert _process_dataframe(None, []) is None
        assert _process_dataframe(pd.DataFrame(), []) is None

    def test_limit_columns(self):
        df = pd.DataFrame(
            {"2023": [100], "2022": [110], "2021": [120], "2020": [130]},
            index=["Revenue"],
        )
        # Limit to 2 columns
        result = _process_dataframe(df, keep_rows=["Revenue"], limit_columns=2)
        assert len(result) == 2
        assert "2023" in result
        assert "2022" in result
        assert "2021" not in result

    def test_filter_rows(self):
        df = pd.DataFrame({"A": [1, 2, 3]}, index=["Revenue", "IgnoreMe", "Profit"])
        result = _process_dataframe(df, keep_rows=["Revenue", "Profit"])
        # Result format is {col: {row: val}}
        assert "Revenue" in result["A"]
        assert "Profit" in result["A"]
        assert "IgnoreMe" not in result["A"]

    def test_filter_rows_keeps_order_if_possible(self):
        # The function creates a new df with .loc[valid_indices].
        # valid_indices are gathered by iterating over df.index and checking if in keep_rows.
        # So it preserves original DF order, but filters out non-kept rows.
        df = pd.DataFrame({"A": [1, 2, 3]}, index=["Revenue", "IgnoreMe", "Profit"])
        result = _process_dataframe(df, keep_rows=["Profit", "Revenue"])
        keys = list(result["A"].keys())
        # Should follow df order: Revenue, then Profit
        assert keys == ["Revenue", "Profit"]


class TestGetEarningsData:
    def test_extracts_net_income(self):
        df_annual = pd.DataFrame({"2023": [1000], "2022": [800]}, index=["Net Income"])
        df_quarterly = pd.DataFrame({"Q1": [200], "Q2": [250]}, index=["Net Income"])

        result = _get_earnings_data(df_annual, df_quarterly)

        assert result["annual"] is not None
        assert result["quarterly"] is not None
        assert result["annual"]["2023"]["Net Income"] == 1000

    def test_missing_net_income(self):
        df = pd.DataFrame({"A": [1]}, index=["Revenue"])
        result = _get_earnings_data(df, df)
        assert result["annual"] is None
        assert result["quarterly"] is None

    def test_none_inputs(self):
        result = _get_earnings_data(None, None)
        assert result["annual"] is None
        assert result["quarterly"] is None
