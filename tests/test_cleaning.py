"""
test_cleaning.py — Unit tests for data cleaning functions.
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.airbnb_pipeline.clean import clean_price, convert_tf_to_bool, parse_date


# ─────────────────────────────── clean_price ───────────────────────────────


class TestCleanPrice:
    """Tests for the clean_price function."""

    def test_simple_dollar_amount(self):
        assert clean_price("$1,234.56") == 1234.56

    def test_no_currency_symbol(self):
        assert clean_price("1234.56") == 1234.56

    def test_with_spaces(self):
        assert clean_price("  $500.00  ") == 500.0

    def test_integer_string(self):
        assert clean_price("$100") == 100.0

    def test_numeric_input(self):
        assert clean_price(99.99) == 99.99

    def test_integer_input(self):
        assert clean_price(100) == 100.0

    def test_none_returns_none(self):
        assert clean_price(None) is None

    def test_nan_returns_none(self):
        assert clean_price(float("nan")) is None

    def test_empty_string_returns_none(self):
        assert clean_price("") is None

    def test_non_numeric_string_returns_none(self):
        assert clean_price("abc") is None

    def test_zero_price(self):
        assert clean_price("$0.00") == 0.0

    def test_large_price(self):
        assert clean_price("$999,999.99") == 999999.99

    def test_thai_baht_symbol(self):
        # Should strip any non-numeric chars
        assert clean_price("฿1,500") == 1500.0


# ─────────────────────────────── convert_tf_to_bool ───────────────────────────────


class TestConvertTfToBool:
    """Tests for t/f to boolean conversion."""

    def test_t_returns_true(self):
        assert convert_tf_to_bool("t") is True

    def test_f_returns_false(self):
        assert convert_tf_to_bool("f") is False

    def test_true_string(self):
        assert convert_tf_to_bool("true") is True

    def test_false_string(self):
        assert convert_tf_to_bool("false") is False

    def test_uppercase_T(self):
        assert convert_tf_to_bool("T") is True

    def test_one_returns_true(self):
        assert convert_tf_to_bool("1") is True

    def test_zero_returns_false(self):
        assert convert_tf_to_bool("0") is False

    def test_none_returns_none(self):
        assert convert_tf_to_bool(None) is None

    def test_nan_returns_none(self):
        assert convert_tf_to_bool(float("nan")) is None

    def test_unexpected_value_returns_none(self):
        assert convert_tf_to_bool("maybe") is None

    def test_whitespace_handled(self):
        assert convert_tf_to_bool("  t  ") is True


# ─────────────────────────────── parse_date ───────────────────────────────


class TestParseDate:
    """Tests for date parsing."""

    def test_standard_date(self):
        result = parse_date("2023-01-15")
        assert result == pd.Timestamp("2023-01-15")

    def test_slash_format(self):
        result = parse_date("01/15/2023")
        assert result is not None
        assert result.year == 2023

    def test_none_returns_none(self):
        assert parse_date(None) is None

    def test_empty_string_returns_none(self):
        assert parse_date("") is None

    def test_invalid_date_returns_none(self):
        result = parse_date("not-a-date")
        assert result is None or pd.isna(result)

    def test_nan_returns_none(self):
        assert parse_date(float("nan")) is None
