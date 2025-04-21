# tests/test_transform.py
"""
Unit tests for the utils.transform module.
"""

import logging
import pytest
import pandas as pd
from datetime import timezone
from unittest.mock import patch
from freezegun import freeze_time

# Assuming utils is importable from the project root
from utils import transform
from utils.constants import (
    FINAL_SCHEMA_TYPE_MAPPING,
    REQUIRED_COLUMNS,
    USD_TO_IDR_RATE,
)

# --- Tests for Cleaning Functions ---


@pytest.mark.parametrize(
    "price_str, expected",
    [
        ("$19.99", 19.99),
        (" $ 1,234.56 ", 1234.56),
        ("Price Unavailable", None),
        ("price unavailable", None),
        (None, None),
        ("Invalid", None),
        ("10.50", 10.50),
        ("$", None), # Only symbol
    ],
)
def test_clean_price(price_str, expected):
    assert transform.clean_price(price_str) == expected


@pytest.mark.parametrize(
    "rating_str, expected",
    [
        ("⭐ 4.5 / 5", 4.5),
        ("Rating: 3.8 stars", 3.8),
        (" 4 ", 4.0),
        ("4.2", 4.2),
        ("Not Rated", None),
        ("invalid rating", None),
        (None, None),
        ("No rating", None),
        ("5/5", 5.0), # Simple fraction format
    ],
)
def test_clean_rating(rating_str, expected):
    assert transform.clean_rating(rating_str) == expected


@pytest.mark.parametrize(
    "colors_str, expected",
    [
        ("Available in 3 Colors", 3),
        (" 5 colors ", 5),
        ("1 Color", 1),
        ("Single color", None), # No number
        ("Color: Blue", None), # No number
        (None, None),
        ("Invalid", None),
    ],
)
def test_clean_colors(colors_str, expected):
    assert transform.clean_colors(colors_str) == expected


# --- Tests for transform_data ---

# Sample data covering various scenarios
SAMPLE_RAW_DATA = [
    # Valid full record
    {
        "title": " T-shirt 1 ",
        "price": "$10.00",
        "rating": "⭐ 4.5 / 5",
        "colors": "3 colors",
        "size": " M ",
        "gender": " Men",
        "image_url": "url1",
    },
    # Valid record, different values
    {
        "title": "Pants 2",
        "price": "$25.50",
        "rating": "⭐ 3.8 / 5",
        "colors": "5 colors",
        "size": "L",
        "gender": "Women",
        "image_url": "url2",
    },
    # Duplicate of first record (should be removed)
    {
        "title": " T-shirt 1 ",
        "price": "$10.00",
        "rating": "⭐ 4.5 / 5",
        "colors": "3 colors",
        "size": " M ",
        "gender": " Men",
        "image_url": "url1",
    },
    # Price unavailable (should become None, potentially dropped if required)
    {
        "title": "Jacket 3",
        "price": "Price Unavailable",
        "rating": "⭐ 4.0 / 5",
        "colors": "2 colors",
        "size": "S",
        "gender": "Unisex",
        "image_url": "url3",
    },
    # Unknown Product (should be filtered out)
    {
        "title": "Unknown Product",
        "price": "$50.00",
        "rating": "⭐ 5.0 / 5",
        "colors": "1 Color",
        "size": "XL",
        "gender": "Men",
        "image_url": "url4",
    },
    # Rating not rated (should become None, potentially dropped if required)
    {
        "title": "Shoes 5",
        "price": "$120.00",
        "rating": "Not Rated",
        "colors": "4 colors",
        "size": "M",
        "gender": "Women",
        "image_url": "url5",
    },
    # Colors None (should become None, potentially dropped if required)
    {
        "title": "Hat 6",
        "price": "$15.00",
        "rating": "⭐ 4.2 / 5",
        "colors": None,
        "size": "OS",
        "gender": "Unisex",
        "image_url": "url6",
    },
    # Size None (should become None, potentially dropped if required)
    {
        "title": "Belt 7",
        "price": "$30.00",
        "rating": "⭐ 4.9 / 5",
        "colors": "1 color",
        "size": None,
        "gender": "Unisex",
        "image_url": "url7",
    },
    # Image URL None (should become None, potentially dropped if required)
    {
        "title": "Item 9 No Img",
        "price": "$40.00",
        "rating": "⭐ 4.0 / 5",
        "colors": "1 color",
        "size": "M",
        "gender": "Women",
        "image_url": None,
    },
    # Complete Item to check final schema
    {
        "title": "Complete Item 8",
        "price": "$75.00",
        "rating": "⭐ 4.1 / 5",
        "colors": "2 colors",
        "size": "L",
        "gender": "Men",
        "image_url": "url8",
    },
]

# Define expected output based on SAMPLE_RAW_DATA and transformation logic
# Assuming REQUIRED_COLUMNS = ['title', 'price', 'image_url'] for this example
# Rows with None in these columns after cleaning will be dropped.
# Also, 'Unknown Product' is filtered, and duplicates are removed.

# Expected intermediate values after cleaning/parsing:
# T-shirt 1: price=10.0, rating=4.5, colors=3
# Pants 2: price=25.5, rating=3.8, colors=5
# Jacket 3: price=None, rating=4.0, colors=2 -> DROPPED (if price required)
# Shoes 5: price=120.0, rating=None, colors=4 -> DROPPED (if rating required)
# Hat 6: price=15.0, rating=4.2, colors=None -> DROPPED (if colors required)
# Belt 7: price=30.0, rating=4.9, colors=1, size=None -> DROPPED (if size required)
# Item 9 No Img: price=40.0, rating=4.0, colors=1, image_url=None -> DROPPED (if image_url required)
# Complete Item 8: price=75.0, rating=4.1, colors=2

# Let's assume REQUIRED_COLUMNS = ['title', 'price', 'image_url', 'size', 'gender']
# Based on this, expected rows to survive:
# T-shirt 1, Pants 2, Complete Item 8

EXPECTED_TITLES = ["T-shirt 1", "Pants 2", "Complete Item 8"]
EXPECTED_PRICES_IDR = [
    10.00 * USD_TO_IDR_RATE,
    25.50 * USD_TO_IDR_RATE,
    75.00 * USD_TO_IDR_RATE,
]
EXPECTED_RATINGS = [4.5, 3.8, 4.1]
EXPECTED_COLORS = [3, 5, 2]
EXPECTED_SIZES = ["M", "L", "L"]
EXPECTED_GENDERS = ["Men", "Women", "Men"]
EXPECTED_IMAGE_URLS = ["url1", "url2", "url8"]
EXPECTED_ROW_COUNT = 3
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


@freeze_time("2024-01-15 12:00:00 UTC") # Freeze time for consistent timestamp
def test_transform_data_happy_path_and_cleaning(caplog):
    """Tests the main transform_data function with various cleaning, filtering, and duplication."""
    # Arrange
    caplog.set_level(logging.INFO)
    # Use a copy to avoid modifying the original list during tests
    test_data = [d.copy() for d in SAMPLE_RAW_DATA]

    # Act
    df_transformed = transform.transform_data(test_data)

    # Assert
    assert not df_transformed.empty
    assert len(df_transformed) == EXPECTED_ROW_COUNT
    assert list(df_transformed.columns) == EXPECTED_COLS

    # Check logs for filtering/dropping
    assert "Filtered 1 rows with 'Unknown Product' title." in caplog.text
    # Calculate expected dropped rows (total - unknown - duplicates - final)
    # Initial = 10, Unknown = 1, Duplicate = 1 -> 8 remain before dropna
    # Final = 3, so 8 - 3 = 5 dropped by dropna
    # The exact number depends heavily on REQUIRED_COLUMNS
    # Let's check if the dropna log message exists if rows were dropped
    initial_rows = len(SAMPLE_RAW_DATA)
    if initial_rows - 1 - 1 > EXPECTED_ROW_COUNT: # If (initial - unknown - duplicate) > final
         assert "Removed" in caplog.text and "rows with null values in required columns" in caplog.text
    assert "Removed 1 duplicate rows." in caplog.text
    assert f"Transformation complete. Final rows: {EXPECTED_ROW_COUNT}" in caplog.text

    # Check data content (sort by title for consistent comparison)
    df_transformed = df_transformed.sort_values(by="title").reset_index(drop=True)

    pd.testing.assert_series_equal(
        df_transformed["title"], pd.Series(sorted(EXPECTED_TITLES)), check_names=False
    )
    # Need to sort expected values based on sorted titles
    sort_indices = [EXPECTED_TITLES.index(t) for t in sorted(EXPECTED_TITLES)]
    pd.testing.assert_series_equal(
        df_transformed["price"], pd.Series([EXPECTED_PRICES_IDR[i] for i in sort_indices]), check_dtype=False, check_names=False
    )
    pd.testing.assert_series_equal(
        df_transformed["rating"], pd.Series([EXPECTED_RATINGS[i] for i in sort_indices]), check_dtype=False, check_names=False
    )
    pd.testing.assert_series_equal(
        df_transformed["colors"], pd.Series([EXPECTED_COLORS[i] for i in sort_indices]), check_dtype=False, check_names=False
    )
    pd.testing.assert_series_equal(
        df_transformed["size"], pd.Series([EXPECTED_SIZES[i] for i in sort_indices]), check_names=False
    )
    pd.testing.assert_series_equal(
        df_transformed["gender"], pd.Series([EXPECTED_GENDERS[i] for i in sort_indices]), check_names=False
    )
    pd.testing.assert_series_equal(
        df_transformed["image_url"], pd.Series([EXPECTED_IMAGE_URLS[i] for i in sort_indices]), check_names=False
    )

    # Check timestamp (should be consistent due to freeze_time)
    expected_timestamp = pd.Timestamp("2024-01-15 12:00:00+0000", tz="UTC").tz_convert("Asia/Jakarta")
    assert all(df_transformed["timestamp"] == expected_timestamp)

    # Check final dtypes
    for col, dtype in FINAL_SCHEMA_TYPE_MAPPING.items():
        assert df_transformed[col].dtype == dtype
    assert pd.api.types.is_datetime64_any_dtype(df_transformed["timestamp"])
    assert df_transformed["timestamp"].dt.tz == timezone(pd.Timedelta(hours=7)) # Asia/Jakarta is UTC+7


def test_transform_data_empty_input(caplog):
    """Tests transform_data with an empty input list."""
    # Arrange
    caplog.set_level(logging.WARNING)

    # Act
    df_transformed = transform.transform_data([])

    # Assert
    assert df_transformed.empty
    assert "Received empty list for transformation." in caplog.text


def test_transform_data_all_invalid_input(caplog):
    """Tests transform_data where all input rows are filtered out or invalid."""
    # Arrange
    test_data = [
        {"title": "Unknown Product", "price": "$10", "rating": "4", "colors": "1", "size": "S", "gender": "M", "image_url": "url"},
        {"title": "Valid Title But Missing Required", "price": None, "rating": "4", "colors": "1", "size": "S", "gender": "M", "image_url": "url"}, # Assume price is required
    ]
    # Modify REQUIRED_COLUMNS temporarily for this test if needed, or ensure price is required
    original_required = transform.REQUIRED_COLUMNS
    transform.REQUIRED_COLUMNS = ['title', 'price'] # Ensure price causes dropna
    caplog.set_level(logging.INFO)

    # Act
    df_transformed = transform.transform_data(test_data)

    # Assert
    assert df_transformed.empty
    assert "Filtered 1 rows with 'Unknown Product' title." in caplog.text
    assert "Removed 1 rows with null values in required columns" in caplog.text # The second row
    assert "Transformation complete. Final rows: 0" in caplog.text

    # Restore original REQUIRED_COLUMNS
    transform.REQUIRED_COLUMNS = original_required


@patch("utils.transform._prepare_final_schema")
def test_transform_data_final_schema_failure(mock_prepare_final, caplog):
    """Tests transform_data when _prepare_final_schema fails (returns empty)."""
    # Arrange
    mock_prepare_final.return_value = pd.DataFrame() # Simulate failure
    test_data = [SAMPLE_RAW_DATA[0].copy()] # Use one valid record
    caplog.set_level(logging.ERROR)

    # Act
    df_transformed = transform.transform_data(test_data)

    # Assert
    assert df_transformed.empty
    assert "Failed during final schema preparation." in caplog.text
    mock_prepare_final.assert_called_once()


@patch("pandas.Timestamp", side_effect=Exception("Timezone DB not found"))
def test_transform_data_timestamp_timezone_error(mock_timestamp, caplog):
    """Tests fallback to UTC if timezone conversion fails."""
    # Arrange
    test_data = [SAMPLE_RAW_DATA[0].copy()] # Use one valid record
    caplog.set_level(logging.WARNING)

    # Act
    df_transformed = transform.transform_data(test_data)

    # Assert
    assert not df_transformed.empty
    assert "Failed to convert timezone to Asia/Jakarta" in caplog.text
    assert "Using UTC." in caplog.text
    assert pd.api.types.is_datetime64_any_dtype(df_transformed["timestamp"])
    assert df_transformed["timestamp"].dt.tz == timezone.utc


def test_transform_data_key_error(caplog):
    """Tests transform_data when input dict is missing an expected key."""
    # Arrange
    # Missing 'price' which is used early in _initial_clean_and_parse
    test_data = [
        {
            "title": " T-shirt 1 ",
            # "price": "$10.00", # Missing
            "rating": "⭐ 4.5 / 5",
            "colors": "3 colors",
            "size": " M ",
            "gender": " Men",
            "image_url": "url1",
        }
    ]
    caplog.set_level(logging.ERROR)

    # Act
    df_transformed = transform.transform_data(test_data)

    # Assert
    assert df_transformed.empty
    assert "Missing expected key during transformation: 'price'" in caplog.text


@patch("utils.transform._apply_business_logic", side_effect=Exception("Unexpected calculation error"))
def test_transform_data_unexpected_exception(mock_apply_logic, caplog):
    """Tests graceful handling of unexpected exceptions during transformation steps."""
    # Arrange
    test_data = [SAMPLE_RAW_DATA[0].copy()] # Use one valid record
    caplog.set_level(logging.ERROR)

    # Act
    df_transformed = transform.transform_data(test_data)

    # Assert
    assert df_transformed.empty
    assert "An unexpected error occurred during data transformation" in caplog.text
    assert "Unexpected calculation error" in caplog.text