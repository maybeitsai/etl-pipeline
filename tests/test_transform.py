# tests/test_transform.py
"""
Unit tests for the transform module (utils/transform.py).
"""
import logging
from datetime import timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from utils.constants import (
    FINAL_SCHEMA_TYPE_MAPPING,
    REQUIRED_COLUMNS,
    USD_TO_IDR_RATE,
)
from utils.transform import (
    _add_timestamp,
    _apply_business_logic,
    _filter_invalid_rows,
    _initial_clean_and_parse,
    _prepare_final_schema,
    _remove_nulls_and_duplicates,
    clean_colors,
    clean_price,
    clean_rating,
    transform_data,
)

# --- Test Data ---
SAMPLE_RAW_DATA = [
    {
        "title": " T-shirt 1 ",
        "price": "$10.00",
        "rating": "⭐ 4.5 / 5",
        "colors": "3 colors",
        "size": " M ",
        "gender": " Men",
        "image_url": "url1",
    },
    {
        "title": "Pants 2",
        "price": "$25.50",
        "rating": "⭐ 3.8 / 5",
        "colors": "5 colors",
        "size": "L",
        "gender": "Women",
        "image_url": "url2",
    },
    {  # Duplicate of first item
        "title": " T-shirt 1 ",
        "price": "$10.00",
        "rating": "⭐ 4.5 / 5",
        "colors": "3 colors",
        "size": " M ",
        "gender": " Men",
        "image_url": "url1",
    },
    {  # Price unavailable
        "title": "Jacket 3",
        "price": "Price Unavailable",
        "rating": "⭐ 4.0 / 5",
        "colors": "2 colors",
        "size": "S",
        "gender": "Unisex",
        "image_url": "url3",
    },
    {  # Unknown Product title
        "title": "Unknown Product",
        "price": "$50.00",
        "rating": "⭐ 5.0 / 5",
        "colors": "1 Color",
        "size": "XL",
        "gender": "Men",
        "image_url": "url4",
    },
    {  # Not Rated
        "title": "Shoes 5",
        "price": "$120.00",
        "rating": "Not Rated",
        "colors": "4 colors",
        "size": "M",
        "gender": "Women",
        "image_url": "url5",
    },
    {  # Null colors
        "title": "Hat 6",
        "price": "$15.00",
        "rating": "⭐ 4.2 / 5",
        "colors": None,
        "size": "OS",
        "gender": "Unisex",
        "image_url": "url6",
    },
    {  # Null size (required column)
        "title": "Belt 7",
        "price": "$30.00",
        "rating": "⭐ 4.9 / 5",
        "colors": "1 color",
        "size": None,
        "gender": "Unisex",
        "image_url": "url7",
    },
    {  # Complete Item
        "title": "Complete Item 8",
        "price": "$75.00",
        "rating": "⭐ 4.1 / 5",
        "colors": "2 colors",
        "size": "L",
        "gender": "Men",
        "image_url": "url8",
    },
    {  # Null image_url (required column)
        "title": "Item 9 No Img",
        "price": "$40.00",
        "rating": "⭐ 4.0 / 5",
        "colors": "1 color",
        "size": "M",
        "gender": "Women",
        "image_url": None,
    },
    {  # Invalid price format
        "title": "Item 10 Invalid Price",
        "price": "Free",
        "rating": "4.3",
        "colors": "1 color",
        "size": "S",
        "gender": "Men",
        "image_url": "url10",
    },
    {  # Invalid rating format
        "title": "Item 11 Invalid Rating",
        "price": "$19.99",
        "rating": "Good",
        "colors": "2 colors",
        "size": "M",
        "gender": "Women",
        "image_url": "url11",
    },
    {  # Invalid colors format
        "title": "Item 12 Invalid Colors",
        "price": "$29.99",
        "rating": "4.6",
        "colors": "Multiple shades",
        "size": "L",
        "gender": "Unisex",
        "image_url": "url12",
    },
    {  # Case variation for Unknown Product
        "title": "unknown product 13",
        "price": "$5.00",
        "rating": "3.0",
        "colors": "1 color",
        "size": "XS",
        "gender": "Kids",
        "image_url": "url13",
    },
]

EXPECTED_COLS = [
    "title",
    "price",
    "rating",
    "colors",
    "size",
    "gender",
    "image_url",
    "timestamp",
]


# --- Tests for Cleaning Functions ---
@pytest.mark.parametrize(
    "price_str, expected",
    [
        ("$10.00", 10.0),
        (" $ 25.50 ", 25.5),
        ("$1,200.99", 1200.99),
        ("99.9", 99.9),
        ("Price Unavailable", None),
        ("unavailable", None),
        (None, None),
        ("", None),
        ("abc", None),
        (100, None),  # Handles unexpected type input
    ],
)
def test_clean_price(price_str, expected, caplog):
    """Test clean_price function with various inputs."""
    with caplog.at_level(logging.DEBUG):
        assert clean_price(price_str) == expected
        if expected is None and price_str not in [
            None,
            "Price Unavailable",
            "unavailable",
            "",
        ]:
            assert f"Could not parse price: '{price_str}'" in caplog.text


@pytest.mark.parametrize(
    "rating_str, expected",
    [
        ("⭐ 4.5 / 5", 4.5),
        (" 3.8 / 5 stars ", 3.8),
        ("Rating: 4.0", 4.0),
        ("5/5", 5.0),
        ("4", 4.0),
        (" 4.2 ", 4.2),
        ("Not Rated", None),
        ("invalid rating", None),
        (None, None),
        ("", None),
        ("abc", None),
        ("No rating available", None),
        ("4/ stars", 4.0),
        (4.5, None),
    ],
)
def test_clean_rating(rating_str, expected, caplog):
    """Test clean_rating function with various inputs."""
    with caplog.at_level(logging.DEBUG):
        assert clean_rating(rating_str) == expected
        if expected is None and rating_str not in [
            None,
            "Not Rated",
            "invalid rating",
            "",
        ]:
            assert f"Could not parse rating: '{rating_str}'" in caplog.text


@pytest.mark.parametrize(
    "colors_str, expected",
    [
        ("3 colors", 3),
        ("Available in 5 Colors", 5),
        (" 1 Color ", 1),
        ("10", 10),
        (None, None),
        ("", None),
        ("Multiple colors", None),
        ("abc", None),
        (3, None),
    ],
)
# tests/test_transform.py - dalam test_clean_colors
def test_clean_colors(colors_str, expected, caplog):
    """Test clean_colors function with various inputs."""
    with caplog.at_level(logging.DEBUG):
        assert clean_colors(colors_str) == expected
        if expected is None and colors_str is not None and colors_str != "":
            # --- TAMBAHKAN PENGECEKAN INI ---
            if isinstance(colors_str, str):
                if "color" in colors_str.lower() and not any(
                    char.isdigit() for char in colors_str
                ):
                    assert (
                        f"Found 'color' text but no number in: '{colors_str}'"
                        in caplog.text
                    )
                # --- Pindahkan else ini agar sejajar dengan if "color" ---
                elif not (
                    "color" in colors_str.lower()
                    and any(char.isdigit() for char in colors_str)
                ):  # Hindari log duplikat jika sudah ketemu angka
                    assert f"Could not parse colors: '{colors_str}'" in caplog.text
            # --- Tambahkan else untuk handle input bukan string (seperti int) ---
            else:
                assert f"Could not parse colors: '{colors_str}'" in caplog.text


@patch("utils.transform.re.search")
def test_clean_colors_triggers_value_error(mock_re_search, caplog):
    """
    Test clean_colors handles ValueError raised during re.search.
    """
    # Konfigurasi mock untuk memunculkan ValueError saat dipanggil
    mock_re_search.side_effect = ValueError("Forced ValueError from regex")

    input_str = "this is a valid string input"
    with caplog.at_level(logging.DEBUG):
        result = clean_colors(input_str)

    assert result is None
    # Pastikan log berasal dari blok except
    assert f"Could not parse colors: '{input_str}'." in caplog.text
    # Pastikan mock dipanggil
    mock_re_search.assert_called_once_with(r"(\d+)", input_str)


# Tes untuk memicu TypeError di dalam blok try
@patch("utils.transform.re.search")
def test_clean_colors_triggers_type_error(mock_re_search, caplog):
    """
    Test clean_colors handles TypeError raised during re.search.
    """
    # Konfigurasi mock untuk memunculkan TypeError saat dipanggil
    mock_re_search.side_effect = TypeError("Forced TypeError from regex")

    input_str = "another valid string input"

    with caplog.at_level(logging.DEBUG):
        result = clean_colors(input_str)

    assert result is None
    # Pastikan log berasal dari blok except
    assert f"Could not parse colors: '{input_str}'." in caplog.text
    # Pastikan mock dipanggil
    mock_re_search.assert_called_once_with(r"(\d+)", input_str)


# Anda mungkin juga ingin mempertahankan tes asli Anda untuk input non-string (jika belum ada)
def test_clean_colors_non_string_input(caplog):
    """Test clean_colors with non-string input."""
    non_string_input = 12345
    with caplog.at_level(logging.DEBUG):
        result = clean_colors(non_string_input)
    assert result is None
    # Log ini berasal dari pemeriksaan isinstance awal
    assert f"Could not parse colors: '{non_string_input}'." in caplog.text


# --- Tests for Transformation Steps ---


def test_initial_clean_and_parse():
    """Test the initial cleaning and parsing step."""
    df_raw = pd.DataFrame(SAMPLE_RAW_DATA[:2])
    df_cleaned = _initial_clean_and_parse(df_raw.copy())

    assert "cleaned_price_usd" in df_cleaned.columns
    assert "cleaned_rating" in df_cleaned.columns
    assert "cleaned_colors" in df_cleaned.columns

    # Check stripping
    assert df_cleaned.loc[0, "title"] == "T-shirt 1"
    assert df_cleaned.loc[0, "size"] == "M"
    assert df_cleaned.loc[0, "gender"] == "Men"

    # Check cleaning results
    assert df_cleaned.loc[0, "cleaned_price_usd"] == 10.0
    assert df_cleaned.loc[1, "cleaned_price_usd"] == 25.5
    assert df_cleaned.loc[0, "cleaned_rating"] == 4.5
    assert df_cleaned.loc[1, "cleaned_rating"] == 3.8
    assert df_cleaned.loc[0, "cleaned_colors"] == 3
    assert df_cleaned.loc[1, "cleaned_colors"] == 5


def test_initial_clean_and_parse_missing_columns():
    """Test initial cleaning when optional columns are missing."""
    data = [{"price": "$10", "rating": "4", "colors": "2"}]
    df_raw = pd.DataFrame(data)
    df_cleaned = _initial_clean_and_parse(df_raw.copy())
    # Should run without error, applying cleaning to existing columns
    assert df_cleaned.loc[0, "cleaned_price_usd"] == 10.0
    assert df_cleaned.loc[0, "cleaned_rating"] == 4.0
    assert df_cleaned.loc[0, "cleaned_colors"] == 2


@pytest.fixture
def mock_timestamp():
    """Mock the pandas Timestamp function."""
    with patch("pandas.Timestamp") as mock_ts:
        yield mock_ts


@patch("utils.transform.pd.Timestamp.now")
def test_add_timestamp(mock_pd_Timestamp_now):
    """Test adding the timestamp column with timezone conversion."""
    # Define expected timestamps
    now_utc = pd.Timestamp("2023-10-27 10:00:00", tz="UTC")
    now_jakarta = pd.Timestamp(
        "2023-10-27 17:00:00", tz="Asia/Jakarta"
    )  # Explicit Jakarta time

    # Setup the mocks
    mock_timestamp_instance = MagicMock(spec=pd.Timestamp)
    mock_timestamp_instance.tz_convert.return_value = now_jakarta
    mock_pd_Timestamp_now.return_value = mock_timestamp_instance

    # Test function
    df = pd.DataFrame({"col1": [1, 2]})
    df_ts = _add_timestamp(df.copy())

    # Assertions
    assert "timestamp" in df_ts.columns
    # Fill all rows with the same timestamp for comparison
    expected_df = df.copy()
    expected_df["timestamp"] = now_jakarta

    # Compare only timestamp columns with direct equality
    pd.testing.assert_series_equal(df_ts["timestamp"], expected_df["timestamp"])

    # Verify mock calls
    mock_pd_Timestamp_now.assert_called_once_with(tz="UTC")
    mock_timestamp_instance.tz_convert.assert_called_once_with("Asia/Jakarta")


@patch("utils.transform.pd.Timestamp.now")
def test_add_timestamp_tz_conversion_error(mock_pd_Timestamp_now, caplog):
    """Test fallback to UTC if timezone conversion fails."""
    # Define expected timestamps
    now_utc = pd.Timestamp("2023-10-27 10:00:00", tz="UTC")

    # Setup the first mock to throw an error
    mock_ts_instance_error = MagicMock(spec=pd.Timestamp)
    mock_ts_instance_error.tz_convert.side_effect = Exception("TZ database error")

    # Setup side effect to return different values on successive calls
    mock_pd_Timestamp_now.side_effect = [mock_ts_instance_error, now_utc]

    # Test function
    df = pd.DataFrame({"col1": [1, 2]})
    with caplog.at_level(logging.WARNING):
        df_ts = _add_timestamp(df.copy())

    # Assertions
    assert "timestamp" in df_ts.columns
    # Fill all rows with the same timestamp for comparison
    expected_df = df.copy()
    expected_df["timestamp"] = now_utc

    # Compare timestamp columns directly
    pd.testing.assert_series_equal(df_ts["timestamp"], expected_df["timestamp"])

    # Verify log messages
    assert "Failed to convert timezone to Asia/Jakarta" in caplog.text
    assert "Using UTC." in caplog.text

    # Verify mock calls
    assert mock_pd_Timestamp_now.call_count == 2
    mock_pd_Timestamp_now.assert_any_call(tz="UTC")
    mock_ts_instance_error.tz_convert.assert_called_once_with("Asia/Jakarta")


def test_filter_invalid_rows(caplog):
    """Test filtering rows with 'Unknown Product' title."""
    df = pd.DataFrame(
        {
            "title": ["Product A", "Unknown Product", "Product B", "unknown product C"],
            "price": [10, 50, 20, 5],
        }
    )
    with caplog.at_level(logging.INFO):
        df_filtered = _filter_invalid_rows(df.copy())

    expected_df = pd.DataFrame(
        {
            "title": ["Product A", "Product B"],
            "price": [10, 20],
        },
        index=[0, 2],
    )  # Keep original index

    assert_frame_equal(df_filtered, expected_df)
    assert "Filtered 2 rows with 'Unknown Product' title." in caplog.text


def test_filter_invalid_rows_no_invalid():
    """Test filtering when no rows are invalid."""
    df = pd.DataFrame(
        {
            "title": ["Product A", "Product B"],
            "price": [10, 20],
        }
    )
    df_filtered = _filter_invalid_rows(df.copy())
    assert_frame_equal(df_filtered, df)


def test_filter_invalid_rows_empty_df():
    """Test filtering with an empty DataFrame."""
    df = pd.DataFrame({"title": []})
    df_filtered = _filter_invalid_rows(df.copy())
    assert df_filtered.empty


def test_apply_business_logic():
    """Test the currency conversion logic."""
    df = pd.DataFrame({"cleaned_price_usd": [10.0, 25.5, None, 50.0]})
    df_logic = _apply_business_logic(df.copy())

    assert "price_idr" in df_logic.columns
    expected_idr = pd.Series(
        [
            10.0 * USD_TO_IDR_RATE,
            25.5 * USD_TO_IDR_RATE,
            None,  # NaN * rate = NaN
            50.0 * USD_TO_IDR_RATE,
        ],
        name="price_idr",
    )

    assert_series_equal(df_logic["price_idr"], expected_idr, check_dtype=False)


def test_apply_business_logic_empty_df():
    """Test business logic with an empty DataFrame."""
    df = pd.DataFrame({"cleaned_price_usd": []})
    df_logic = _apply_business_logic(df.copy())
    assert "price_idr" in df_logic.columns
    assert df_logic.empty


def test_prepare_final_schema():
    """Test column selection, renaming, and type enforcement."""
    # Create a DataFrame simulating state before this step
    now = pd.Timestamp.now(tz="Asia/Jakarta")
    df_intermediate = pd.DataFrame(
        {
            "title": ["Product A", "Product B"],
            "cleaned_price_usd": [10.0, 20.0],
            "price_idr": [160000.0, 320000.0],
            "cleaned_rating": [4.5, 4.0],
            "cleaned_colors": [3, 1],
            "size": ["M", "S"],
            "gender": ["Men", "Women"],
            "image_url": ["urlA", "urlB"],
            "timestamp": [now, now],
            "extra_col": ["x", "y"],
        }
    )

    df_final = _prepare_final_schema(df_intermediate.copy())

    # Check columns exist and are renamed
    assert sorted(df_final.columns) == sorted(EXPECTED_COLS)
    assert "extra_col" not in df_final.columns
    assert "cleaned_price_usd" not in df_final.columns

    # Check renaming results
    assert df_final["price"].equals(df_intermediate["price_idr"])
    assert df_final["rating"].equals(df_intermediate["cleaned_rating"])
    assert df_final["colors"].equals(df_intermediate["cleaned_colors"])

    # Check final schema has expected column types
    for col, expected_pd_type in FINAL_SCHEMA_TYPE_MAPPING.items():
        if col in df_final.columns:
            actual_dtype = df_final[col].dtype
            if expected_pd_type == str:
                assert pd.api.types.is_string_dtype(
                    actual_dtype
                ) or pd.api.types.is_object_dtype(
                    actual_dtype
                ), f"Column '{col}' expected str, got {actual_dtype}"
            elif expected_pd_type == float:
                assert pd.api.types.is_float_dtype(
                    actual_dtype
                ), f"Column '{col}' expected float, got {actual_dtype}"
            elif isinstance(expected_pd_type, pd.Int64Dtype):
                assert pd.api.types.is_integer_dtype(
                    actual_dtype
                ), f"Column '{col}' expected Int64Dtype compatible, got {actual_dtype}"
            elif col == "timestamp":
                assert pd.api.types.is_datetime64_any_dtype(
                    actual_dtype
                ), f"Column '{col}' expected datetime, got {actual_dtype}"
            else:
                assert pd.api.types.is_dtype_equal(
                    actual_dtype, expected_pd_type
                ), f"Column '{col}' expected {expected_pd_type}, got {actual_dtype}"


def test_prepare_final_schema_int64_dtype_conversion():
    """Test Int64Dtype conversion in _prepare_final_schema."""
    now = pd.Timestamp.now(tz="Asia/Jakarta")
    df_intermediate = pd.DataFrame(
        {
            "title": ["Product A"],
            "price_idr": [10000.0],
            "cleaned_rating": [4.5],
            "cleaned_colors": [3],
            "size": ["M"],
            "gender": ["Men"],
            "image_url": ["urlA"],
            "timestamp": [now],
        }
    )

    # Mock FINAL_SCHEMA_TYPE_MAPPING to ensure colors uses Int64Dtype
    schema_mapping = {
        "title": str,
        "price": float,
        "rating": float,
        "colors": pd.Int64Dtype(),
        "size": str,
        "gender": str,
        "image_url": str,
        "timestamp": "datetime64[ns, Asia/Jakarta]",
    }

    with patch("utils.transform.FINAL_SCHEMA_TYPE_MAPPING", schema_mapping):
        df_final = _prepare_final_schema(df_intermediate.copy())

    # Verify the colors column was converted to Int64Dtype
    assert isinstance(df_final["colors"].dtype, pd.Int64Dtype)


def test_prepare_final_schema_missing_expected_cols(caplog):
    """Test schema preparation when expected columns are missing."""
    df_intermediate = pd.DataFrame(
        {
            "title": ["A"],
            "price_idr": [1000.0],
        }
    )
    with caplog.at_level(logging.ERROR):
        df_final = _prepare_final_schema(df_intermediate.copy())

    assert df_final.empty
    assert "Missing expected columns before final selection" in caplog.text
    assert "'cleaned_rating'" in caplog.text  # Example check


# tests/test_transform.py - dalam test_prepare_final_schema_type_conversion_error
def test_prepare_final_schema_type_conversion_error(caplog):
    """Test schema preparation when type conversion fails."""
    now = pd.Timestamp.now(tz="Asia/Jakarta")
    df_intermediate = pd.DataFrame(
        {
            "title": ["Product A"],
            "price_idr": ["not_a_number"],
            "cleaned_rating": [4.5],
            "cleaned_colors": [3],
            "size": ["M"],
            "gender": ["Men"],
            "image_url": ["urlA"],
            "timestamp": [now],
        }
    )

    with caplog.at_level(logging.ERROR):
        df_final = _prepare_final_schema(df_intermediate.copy())

    assert not df_final.empty
    assert "price" in df_final.columns
    assert "Error during final data type conversion" in caplog.text
    assert pd.api.types.is_object_dtype(df_final["price"].dtype)


def test_prepare_final_schema_incorrect_timestamp_type():
    """Test schema preparation when timestamp column is not datetime."""
    df_intermediate = pd.DataFrame(
        {
            "title": ["Product A"],
            "price_idr": [160000.0],
            "cleaned_rating": [4.5],
            "cleaned_colors": [3],
            "size": ["M"],
            "gender": ["Men"],
            "image_url": ["urlA"],
            "timestamp": ["2023-10-27 10:00:00+07:00"],
        }
    )
    df_final = _prepare_final_schema(df_intermediate.copy())
    assert pd.api.types.is_datetime64_any_dtype(df_final["timestamp"])
    assert df_final.loc[0, "timestamp"] == pd.Timestamp("2023-10-27 10:00:00+07:00")


def test_remove_nulls_and_duplicates(caplog):
    """Test removal of nulls in required columns and duplicates."""
    now = pd.Timestamp.now(tz="Asia/Jakarta")
    df_with_issues = pd.DataFrame(
        {
            "title": ["A", "B", "C", "A", "D", "E"],
            "price": [10.0, 20.0, 30.0, 10.0, 40.0, 50.0],
            "rating": [4, 5, None, 4, 3, 2],
            "colors": [1, 2, 3, 1, 4, 5],
            "size": ["S", "M", None, "S", "L", "M"],
            "gender": ["M", "F", "M", "M", "F", "F"],
            "image_url": ["url1", "url2", "url3", "url1", None, "url5"],
            "timestamp": [now] * 6,
        }
    )
    # Manually define which columns are required for this test context
    # This assumes REQUIRED_COLUMNS = ["title", "size", "gender", "image_url"]
    with patch(
        "utils.transform.REQUIRED_COLUMNS", ["title", "size", "gender", "image_url"]
    ):
        with caplog.at_level(logging.INFO):
            df_cleaned = _remove_nulls_and_duplicates(df_with_issues.copy())

    expected_df = pd.DataFrame(
        {
            "title": ["A", "B", "E"],
            "price": [10.0, 20.0, 50.0],
            "rating": [4.0, 5.0, 2.0],
            "colors": [1, 2, 5],
            "size": ["S", "M", "M"],
            "gender": ["M", "F", "F"],
            "image_url": ["url1", "url2", "url5"],
            "timestamp": [now] * 3,
        },
        index=[0, 1, 5],
    )

    assert_frame_equal(df_cleaned, expected_df)
    assert "Removed 2 rows with null values in required columns" in caplog.text
    assert "Removed 1 duplicate rows." in caplog.text


def test_remove_nulls_and_duplicates_no_issues():
    """Test the function when no nulls or duplicates exist."""
    now = pd.Timestamp.now(tz="Asia/Jakarta")
    df_clean = pd.DataFrame(
        {
            "title": ["A", "B"],
            "price": [10.0, 20.0],
            "rating": [4, 5],
            "colors": [1, 2],
            "size": ["S", "M"],
            "gender": ["M", "F"],
            "image_url": ["url1", "url2"],
            "timestamp": [now] * 2,
        }
    )
    with patch(
        "utils.transform.REQUIRED_COLUMNS", ["title", "size", "gender", "image_url"]
    ):
        df_result = _remove_nulls_and_duplicates(df_clean.copy())
    assert_frame_equal(df_result, df_clean)


def test_remove_nulls_and_duplicates_empty_df():
    """Test the function with an empty DataFrame."""
    df_empty = pd.DataFrame(columns=EXPECTED_COLS)
    df_result = _remove_nulls_and_duplicates(df_empty.copy())
    assert df_result.empty


def test_remove_nulls_and_duplicates_empty_after_na(caplog):
    """Test when DataFrame becomes empty after dropping NAs."""
    now = pd.Timestamp.now(tz="Asia/Jakarta")
    df_all_na = pd.DataFrame(
        {
            "title": ["A", "B"],
            "price": [10.0, 20.0],
            "rating": [4, 5],
            "colors": [1, 2],
            "size": [None, None],
            "gender": ["M", "F"],
            "image_url": ["url1", "url2"],
            "timestamp": [now] * 2,
        }
    )
    with patch("utils.transform.REQUIRED_COLUMNS", ["size"]):
        with caplog.at_level(logging.WARNING):
            df_result = _remove_nulls_and_duplicates(df_all_na.copy())
    assert df_result.empty
    assert "DataFrame empty after removing rows with null values." in caplog.text


# --- Tests for Main Transformation Function ---
@patch(
    "utils.transform._add_timestamp",
    side_effect=lambda df: df.assign(
        timestamp=pd.Timestamp("2023-01-01", tz="Asia/Jakarta")
    ),
)
@patch(
    "utils.transform.REQUIRED_COLUMNS", ["title", "size", "gender", "image_url"]
)  # Define for test consistency
def test_transform_data_success(mock_add_ts, caplog):
    """Test the main transform_data function with valid sample data."""

    # Expected output based on SAMPLE_RAW_DATA and the mocked timestamp/required cols
    # Rows removed due to:
    # - Index 2 (duplicate)
    # - Index 4 (Unknown Product)
    # - Index 7 (Null size)
    # - Index 9 (Null image_url)
    # - Index 13 (unknown product)
    # - Additional rows removed due to NULL in price, rating, or colors

    # The only rows that should remain are those with complete data in all the critical columns
    expected_data = {
        "title": [
            "T-shirt 1",
            "Pants 2",
            "Complete Item 8",
        ],
        "price": [
            10.0 * USD_TO_IDR_RATE,
            25.5 * USD_TO_IDR_RATE,
            75.0 * USD_TO_IDR_RATE,
        ],
        "rating": [4.5, 3.8, 4.1],
        "colors": [3, 5, 2],
        "size": ["M", "L", "L"],
        "gender": [
            "Men",
            "Women",
            "Men",
        ],
        "image_url": [
            "url1",
            "url2",
            "url8",
        ],
        "timestamp": pd.Timestamp("2023-01-01", tz="Asia/Jakarta"),
    }

    # Convert to float where applicable for comparison precision
    expected_data["price"] = [
        float(p) if p is not None else None for p in expected_data["price"]
    ]
    expected_data["rating"] = [
        float(r) if r is not None else None for r in expected_data["rating"]
    ]
    expected_data["colors"] = [
        int(c) if c is not None else None for c in expected_data["colors"]
    ]

    expected_df = pd.DataFrame(expected_data)

    # Set dtypes for the expected DataFrame
    for col, dtype in FINAL_SCHEMA_TYPE_MAPPING.items():
        if col in expected_df.columns:
            # For nullable integer columns, convert while preserving NaN values
            if dtype == pd.Int64Dtype() and expected_df[col].isna().any():
                expected_df[col] = pd.Series(expected_df[col], dtype=pd.Int64Dtype())
            else:
                try:
                    expected_df[col] = expected_df[col].astype(dtype)
                except:
                    pass

    with caplog.at_level(logging.INFO):
        transformed_df = transform_data(SAMPLE_RAW_DATA)

    # Sort by title to ensure consistent order for comparison
    transformed_df = transformed_df.sort_values(by="title").reset_index(drop=True)
    expected_df = expected_df.sort_values(by="title").reset_index(drop=True)

    assert not transformed_df.empty
    assert len(transformed_df) == 3
    assert mock_add_ts.called
    assert "Transformation complete. Final rows: 3" in caplog.text

    # Compare DataFrames with less strict type checking
    for col in expected_df.columns:
        # Verify values match
        pd.testing.assert_series_equal(
            transformed_df[col],
            expected_df[col],
            check_dtype=False,
            check_names=False,
        )


def test_transform_data_empty_input(caplog):
    """Test transform_data with an empty input list."""
    with caplog.at_level(logging.WARNING):
        result_df = transform_data([])
    assert result_df.empty
    assert "Received empty list for transformation." in caplog.text


@patch("utils.transform._filter_invalid_rows", return_value=pd.DataFrame())
def test_transform_data_empty_after_filter(mock_filter, caplog):
    """Test transform_data when filtering results in an empty DataFrame."""
    with caplog.at_level(logging.WARNING):
        result_df = transform_data(SAMPLE_RAW_DATA[:1])
    assert result_df.empty
    mock_filter.assert_called_once()
    assert "DataFrame empty after filtering invalid rows." in caplog.text


@patch("utils.transform._prepare_final_schema", return_value=pd.DataFrame())
def test_transform_data_empty_after_schema_prep(mock_schema, caplog):
    """Test transform_data when schema preparation returns an empty DataFrame."""
    # Ensure the dataframe is not empty *before* schema prep is called
    with patch("utils.transform._filter_invalid_rows", side_effect=lambda df: df):
        with caplog.at_level(logging.ERROR):
            result_df = transform_data(SAMPLE_RAW_DATA[:1])

    assert result_df.empty
    mock_schema.assert_called_once()
    assert (
        "Failed during final schema preparation. DataFrame became empty." in caplog.text
    )


def test_transform_data_key_error(caplog):
    """Test transform_data handling KeyError (e.g., missing column in raw data)."""
    invalid_data = [{"price": "$10"}]
    with caplog.at_level(logging.ERROR):
        result_df = transform_data(invalid_data)
    assert result_df.empty
    assert "Missing expected key during transformation" in caplog.text


# tests/test_transform.py - dalam test_transform_data_unexpected_error
def test_transform_data_unexpected_error(caplog):
    """Test transform_data handling unexpected exceptions."""
    # --- PERBAIKI MOCK SIDE EFFECT ---
    # Efek samping: error pada panggilan pertama, DF kosong pada panggilan kedua
    mock_effects = [Exception("Test unexpected error"), pd.DataFrame()]
    with patch("utils.transform.pd.DataFrame", side_effect=mock_effects) as mock_df:
        # --- AKHIR PERUBAHAN MOCK ---
        with caplog.at_level(logging.ERROR):
            result_df = transform_data(SAMPLE_RAW_DATA[:1])

    # Hasilnya harus DataFrame kosong yang dikembalikan dari blok except
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.empty
    assert "An unexpected error occurred during data transformation" in caplog.text
    assert "Test unexpected error" in caplog.text
    assert mock_df.call_count == 2


def test_transform_data_empty_after_nulls_duplicates(caplog):
    """Test when DataFrame becomes empty after removing nulls and duplicates."""
    # Mock input data where all rows have nulls in required columns
    sample_data = [
        {
            "title": "Belt 7",
            "price": "$30.00",
            "rating": "⭐ 4.9 / 5",
            "colors": "1 color",
            "size": None,
            "gender": "Unisex",
            "image_url": "url7",
        },
        {
            "title": "Hat 8",
            "price": "$15.00",
            "rating": "⭐ 4.2 / 5",
            "colors": "2 colors",
            "size": None,
            "gender": "Unisex",
            "image_url": "url8",
        },
    ]

    # Skip filtering for this test to ensure we reach the nulls/duplicates step
    with patch("utils.transform._filter_invalid_rows", side_effect=lambda df: df):
        with caplog.at_level(logging.WARNING):
            result_df = transform_data(sample_data)

    assert result_df.empty
    assert "DataFrame empty after removing nulls/duplicates." in caplog.text
