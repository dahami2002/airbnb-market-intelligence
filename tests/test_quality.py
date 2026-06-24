"""
test_quality.py — Unit tests for data quality check functions.
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.airbnb_pipeline.quality import (
    check_boolean,
    check_no_duplicates,
    check_not_null,
    check_positive,
    check_range,
    check_valid_date,
)


# ─────────────────────────────── check_not_null ───────────────────────────────


class TestCheckNotNull:
    """Tests for the not-null quality check."""

    def test_all_present(self):
        df = pd.DataFrame({"id": [1, 2, 3]})
        result = check_not_null(df, "id", "test_table")
        assert result["status"] == "PASS"
        assert result["failed_rows"] == 0

    def test_some_nulls(self):
        df = pd.DataFrame({"id": [1, None, 3]})
        result = check_not_null(df, "id", "test_table")
        assert result["status"] == "FAIL"
        assert result["failed_rows"] == 1

    def test_all_nulls(self):
        df = pd.DataFrame({"id": [None, None, None]})
        result = check_not_null(df, "id", "test_table")
        assert result["status"] == "FAIL"
        assert result["failed_rows"] == 3

    def test_missing_column(self):
        df = pd.DataFrame({"other": [1, 2]})
        result = check_not_null(df, "id", "test_table")
        assert result["status"] == "FAIL"


# ─────────────────────────────── check_no_duplicates ───────────────────────────


class TestCheckNoDuplicates:
    """Tests for the duplicate check."""

    def test_no_duplicates(self):
        df = pd.DataFrame({"id": [1, 2, 3]})
        result = check_no_duplicates(df, "id", "test_table")
        assert result["status"] == "PASS"

    def test_with_duplicates(self):
        df = pd.DataFrame({"id": [1, 2, 2, 3]})
        result = check_no_duplicates(df, "id", "test_table")
        assert result["status"] == "WARN"
        assert result["failed_rows"] == 1

    def test_missing_column(self):
        df = pd.DataFrame({"other": [1, 2]})
        result = check_no_duplicates(df, "id", "test_table")
        assert result["status"] == "FAIL"


# ─────────────────────────────── check_range ───────────────────────────────


class TestCheckRange:
    """Tests for the range check."""

    def test_all_in_range(self):
        df = pd.DataFrame({"lat": [13.7, 13.8, 13.9]})
        result = check_range(df, "lat", "test_table", -90, 90)
        assert result["status"] == "PASS"

    def test_out_of_range(self):
        df = pd.DataFrame({"lat": [13.7, 100.0, 13.9]})
        result = check_range(df, "lat", "test_table", -90, 90)
        assert result["status"] == "FAIL"
        assert result["failed_rows"] == 1

    def test_with_nulls_ignored(self):
        df = pd.DataFrame({"lat": [13.7, None, 13.9]})
        result = check_range(df, "lat", "test_table", -90, 90)
        assert result["status"] == "PASS"  # Nulls are not counted as failures


# ─────────────────────────────── check_positive ───────────────────────────────


class TestCheckPositive:
    """Tests for the positive value check."""

    def test_all_positive(self):
        df = pd.DataFrame({"price": [100, 200, 300]})
        result = check_positive(df, "price", "test_table")
        assert result["status"] == "PASS"

    def test_with_zero(self):
        df = pd.DataFrame({"price": [100, 0, 300]})
        result = check_positive(df, "price", "test_table")
        assert result["status"] == "WARN"
        assert result["failed_rows"] == 1

    def test_with_negative(self):
        df = pd.DataFrame({"price": [100, -50, 300]})
        result = check_positive(df, "price", "test_table")
        assert result["status"] == "WARN"
        assert result["failed_rows"] == 1


# ─────────────────────────────── check_boolean ───────────────────────────────


class TestCheckBoolean:
    """Tests for the boolean check."""

    def test_valid_booleans(self):
        df = pd.DataFrame({"available": [True, False, True]})
        result = check_boolean(df, "available", "test_table")
        assert result["status"] == "PASS"

    def test_with_nulls(self):
        df = pd.DataFrame({"available": [True, None, False]})
        result = check_boolean(df, "available", "test_table")
        assert result["status"] == "PASS"  # Nulls allowed


# ─────────────────────────────── check_valid_date ───────────────────────────────


class TestCheckValidDate:
    """Tests for the date validity check."""

    def test_valid_dates(self):
        df = pd.DataFrame({"date": ["2023-01-01", "2023-06-15"]})
        result = check_valid_date(df, "date", "test_table")
        assert result["status"] == "PASS"

    def test_invalid_dates(self):
        df = pd.DataFrame({"date": ["2023-01-01", "not-a-date"]})
        result = check_valid_date(df, "date", "test_table")
        assert result["status"] == "WARN"
        assert result["failed_rows"] == 1

    def test_missing_column(self):
        df = pd.DataFrame({"other": [1, 2]})
        result = check_valid_date(df, "date", "test_table")
        assert result["status"] == "FAIL"
