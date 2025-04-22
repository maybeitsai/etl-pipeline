# utils/transform.py
"""
Module for transforming the raw extracted product data into a clean, structured format.
"""

import logging
import re
from typing import Dict, List, Optional

import pandas as pd

from utils.constants import (
    FINAL_SCHEMA_TYPE_MAPPING,
    LOG_FORMAT,
    REQUIRED_COLUMNS,
    USD_TO_IDR_RATE,
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


# --- Cleaning Functions ---
def clean_price(price_str: Optional[str]) -> Optional[float]:
    """Cleans the price string (USD) and converts it to float."""
    if not isinstance(price_str, str):
        logging.debug("Could not parse price: '%s'.", price_str)
        return None
    if price_str is None or "unavailable" in price_str.lower():
        return None
    try:
        cleaned_price = re.sub(r"[$,]", "", price_str).strip()
        return float(cleaned_price)
    except (ValueError, TypeError):
        logging.debug("Could not parse price: '%s'.", price_str)
        return None


def clean_rating(rating_str: Optional[str]) -> Optional[float]:
    """Cleans the rating string and extracts the numeric rating as float."""
    if not isinstance(rating_str, str):
        logging.debug("Could not parse rating: '%s'.", rating_str)
        return None
    if (
        rating_str is None
        or "invalid" in rating_str.lower()
        or "not rated" in rating_str.lower()
    ):
        return None
    try:
        match = re.search(r"(\d(\.\d)?)\s*(?:/|\s|$)", rating_str)
        if match:
            return float(match.group(1))
        # Fallback for plain number strings
        return float(rating_str.strip())
    except (ValueError, TypeError):
        logging.debug("Could not parse rating: '%s'.", rating_str)
        return None


def clean_colors(colors_str: Optional[str]) -> Optional[int]:
    """Cleans the colors string and extracts the number of colors as int."""
    if colors_str is None or not isinstance(colors_str, str):
        logging.debug("Could not parse colors: '%s'.", colors_str)
        return None

    try:
        match = re.search(r"(\d+)", colors_str)
        if match:
            return int(match.group(1))

        if "color" in colors_str.lower():
            logging.debug("Found 'color' text but no number in: '%s'.", colors_str)
        else:
            logging.debug("Could not parse colors: '%s'.", colors_str)
        return None
    except (ValueError, TypeError):
        logging.debug("Could not parse colors: '%s'.", colors_str)
        return None


# --- Transformation Steps ---


def _initial_clean_and_parse(df: pd.DataFrame) -> pd.DataFrame:
    """Applies initial string stripping and runs cleaning functions."""
    logging.debug("Applying initial cleaning and parsing.")
    # Strip whitespace from potential string columns
    for col in ["title", "size", "gender", "image_url"]:
        if col in df.columns:
            df[col] = df[col].str.strip()

    df["cleaned_price_usd"] = df["price"].apply(clean_price)
    df["cleaned_rating"] = df["rating"].apply(clean_rating)
    df["cleaned_colors"] = df["colors"].apply(clean_colors)
    return df


def _add_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a timestamp column, converting UTC to Asia/Jakarta."""
    logging.debug("Adding timestamp.")
    try:
        now_utc = pd.Timestamp.now(tz="UTC")
        now_jakarta = now_utc.tz_convert("Asia/Jakarta")
        df["timestamp"] = now_jakarta
    except Exception as tz_error:
        logging.warning(
            "Failed to convert timezone to Asia/Jakarta: %s. Using UTC.", tz_error
        )
        now_utc = pd.Timestamp.now(tz="UTC")
        df["timestamp"] = now_utc
    return df


def _filter_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filters out rows considered invalid (e.g., 'Unknown Product')."""
    initial_rows = len(df)

    # Handle empty DataFrame case
    if df.empty or "title" not in df.columns:
        return df

    # Filter "Unknown Product" titles (case-insensitive)
    unknown_mask = df["title"].str.contains("Unknown Product", na=False, case=False)
    df_filtered = df[~unknown_mask].copy()
    rows_after_unknown = len(df_filtered)
    if initial_rows > rows_after_unknown:
        logging.info(
            "Filtered %d rows with 'Unknown Product' title.",
            initial_rows - rows_after_unknown,
        )
    return df_filtered


def _apply_business_logic(df: pd.DataFrame) -> pd.DataFrame:
    """Applies business rules like currency conversion."""
    logging.debug("Applying business logic (currency conversion).")
    df["price_idr"] = df["cleaned_price_usd"] * USD_TO_IDR_RATE
    return df


def _prepare_final_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Selects, renames columns and enforces final data types."""
    logging.debug("Preparing final schema.")
    column_mapping = {
        "price_idr": "price",
        "cleaned_rating": "rating",
        "cleaned_colors": "colors",
    }
    # Define columns expected at this stage before renaming/selection
    current_expected_cols = [
        "title",
        "price_idr",
        "cleaned_rating",
        "cleaned_colors",
        "size",
        "gender",
        "image_url",
        "timestamp",
    ]
    # Select only the columns needed for the final output
    missing_cols = [col for col in current_expected_cols if col not in df.columns]
    if missing_cols:
        logging.error(
            "Missing expected columns before final selection: %s", missing_cols
        )
        # Return empty DataFrame to signal a critical schema issue
        return pd.DataFrame()

    df_selected = df[current_expected_cols].copy()
    df_renamed = df_selected.rename(columns=column_mapping)

    # Enforce final data types (excluding timestamp, handled separately)
    try:
        # Convert each column individually to maintain nulls properly
        df_typed = df_renamed.copy()
        for col, expected_type in FINAL_SCHEMA_TYPE_MAPPING.items():
            if col in df_typed.columns:
                # Skip timestamp, will handle separately
                if col != "timestamp":
                    # Special handling for nullable integer columns
                    if expected_type == pd.Int64Dtype():
                        df_typed[col] = pd.Series(
                            df_typed[col].values, dtype=pd.Int64Dtype()
                        )
                    else:
                        df_typed[col] = df_typed[col].astype(expected_type)

        # Ensure timestamp is datetime type (it should be, but double-check)
        if (
            "timestamp" in df_typed.columns
            and not pd.api.types.is_datetime64_any_dtype(df_typed["timestamp"])
        ):
            df_typed["timestamp"] = pd.to_datetime(df_typed["timestamp"])

        logging.debug("Final data types enforced.")
        return df_typed

    except (TypeError, ValueError) as e:
        logging.error(
            "Error during final data type conversion for column '%s': %s", col, e
        )
        # Return dataframe before type casting error occurred
        return df_renamed


def _remove_nulls_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Removes rows with nulls in required columns and duplicate rows."""
    if df.empty:
        return df

    initial_rows = len(df)
    # Remove rows with nulls in essential columns defined in constants
    df.dropna(subset=REQUIRED_COLUMNS, inplace=True)
    rows_after_na = len(df)
    if initial_rows > rows_after_na:
        logging.info(
            "Removed %d rows with null values in required columns (%s).",
            initial_rows - rows_after_na,
            REQUIRED_COLUMNS,
        )

    if df.empty:
        logging.warning("DataFrame empty after removing rows with null values.")
        return df

    # Remove duplicate rows based on all columns
    rows_before_duplicates = len(df)
    df.drop_duplicates(inplace=True, keep="first")
    rows_after_duplicates = len(df)
    if rows_before_duplicates > rows_after_duplicates:
        logging.info(
            "Removed %d duplicate rows.", rows_before_duplicates - rows_after_duplicates
        )

    return df


# --- Main Transformation Function ---
def transform_data(extracted_data: List[Dict[str, Optional[str]]]) -> pd.DataFrame:
    """
    Transforms extracted product data into a cleaned and structured DataFrame.

    Args:
        extracted_data: List of product dictionaries from extraction.

    Returns:
        A cleaned, transformed Pandas DataFrame. Returns an empty DataFrame
        if input is empty or transformation fails critically.
    """
    if not extracted_data:
        logging.warning(
            "Received empty list for transformation. Returning empty DataFrame."
        )
        return pd.DataFrame()

    logging.info("Starting transformation for %d raw records.", len(extracted_data))
    initial_rows = len(extracted_data)

    try:
        df = pd.DataFrame(extracted_data)

        # Apply transformation steps sequentially
        df = _initial_clean_and_parse(df)
        df = _add_timestamp(df)
        df = _filter_invalid_rows(df)
        if df.empty:
            logging.warning("DataFrame empty after filtering invalid rows.")
            return df

        df = _apply_business_logic(df)

        # 5. Hapus nulls & duplikat SEBELUM konversi tipe final
        df_clean_intermediate = _remove_nulls_and_duplicates(df.copy())
        if df_clean_intermediate.empty:
            logging.warning("DataFrame empty after removing nulls/duplicates.")
            return df_clean_intermediate

        # 6. Siapkan skema final (termasuk konversi tipe)
        df_final_schema = _prepare_final_schema(df_clean_intermediate)

        if df_final_schema.empty and not df_clean_intermediate.empty:
            # Log error spesifik jika schema prep yang menyebabkan kosong
            logging.error(
                "Failed during final schema preparation. DataFrame became empty."
            )
            # Kita bisa return df_final_schema (yang kosong) atau pd.DataFrame() baru
            return df_final_schema

        # df_clean sudah merupakan hasil akhir sekarang
        df_clean = df_final_schema

        final_rows = len(df_clean)
        # Hitung initial_rows dari df *setelah* filtering unknown product, sebelum dropna/dropdup
        initial_rows_for_logging = len(extracted_data)
        logging.info(
            "Transformation complete. Final rows: %d (processed %d raw records).",
            final_rows,
            initial_rows_for_logging,
        )
        if not df_clean.empty:
            logging.debug("Final DataFrame head:\n%s", df_clean.head().to_string())
            logging.debug("Final data types:\n%s", df_clean.dtypes)

        return df_clean

    except KeyError as e:
        logging.error(
            "Missing expected key during transformation: %s. Check extract keys.",
            e,
        )
        return pd.DataFrame()
    except Exception as e:
        logging.error(
            "An unexpected error occurred during data transformation: %s",
            e,
            exc_info=True,
        )
        return pd.DataFrame()


# --- Example Usage ---
# if __name__ == "__main__":
#     # Sample data mimicking extract output
#     sample_extracted_data = [
#         {
#             "title": " T-shirt 1 ",
#             "price": "$10.00",
#             "rating": "⭐ 4.5 / 5",
#             "colors": "3 colors",
#             "size": " M ",
#             "gender": " Men",
#             "image_url": "url1",
#         },
#         {
#             "title": "Pants 2",
#             "price": "$25.50",
#             "rating": "⭐ 3.8 / 5",
#             "colors": "5 colors",
#             "size": "L",
#             "gender": "Women",
#             "image_url": "url2",
#         },
#         {
#             "title": " T-shirt 1 ",
#             "price": "$10.00",
#             "rating": "⭐ 4.5 / 5",
#             "colors": "3 colors",
#             "size": " M ",
#             "gender": " Men",
#             "image_url": "url1",
#         },
#         {
#             "title": "Jacket 3",
#             "price": "Price Unavailable",
#             "rating": "⭐ 4.0 / 5",
#             "colors": "2 colors",
#             "size": "S",
#             "gender": "Unisex",
#             "image_url": "url3",
#         },
#         {
#             "title": "Unknown Product",
#             "price": "$50.00",
#             "rating": "⭐ 5.0 / 5",
#             "colors": "1 Color",
#             "size": "XL",
#             "gender": "Men",
#             "image_url": "url4",
#         },
#         {
#             "title": "Shoes 5",
#             "price": "$120.00",
#             "rating": "Not Rated",
#             "colors": "4 colors",
#             "size": "M",
#             "gender": "Women",
#             "image_url": "url5",
#         },
#         {
#             "title": "Hat 6",
#             "price": "$15.00",
#             "rating": "⭐ 4.2 / 5",
#             "colors": None,
#             "size": "OS",
#             "gender": "Unisex",
#             "image_url": "url6",
#         },
#         {
#             "title": "Belt 7",
#             "price": "$30.00",
#             "rating": "⭐ 4.9 / 5",
#             "colors": "1 color",
#             "size": None,
#             "gender": "Unisex",
#             "image_url": "url7",
#         },
#         {
#             "title": "Complete Item 8",
#             "price": "$75.00",
#             "rating": "⭐ 4.1 / 5",
#             "colors": "2 colors",
#             "size": "L",
#             "gender": "Men",
#             "image_url": "url8",
#         },
#         {
#             "title": "Item 9 No Img",
#             "price": "$40.00",
#             "rating": "⭐ 4.0 / 5",
#             "colors": "1 color",
#             "size": "M",
#             "gender": "Women",
#             "image_url": None,
#         },
#     ]
#     print("\n--- Transformation Example ---")
#     transformed_df = transform_data(sample_extracted_data)

#     if not transformed_df.empty:
#         print("\n--- Transformation Output ---")
#         print(transformed_df.to_string())
#         print("\nFinal Data types:")
#         print(transformed_df.dtypes)
#     else:
#         print("\nTransformation resulted in an empty DataFrame.")
