# utils/transform.py
"""
Module for transforming the raw extracted product data.
Includes cleaning individual fields, applying business rules (like currency conversion),
filtering invalid data, handling duplicates, and enforcing final schema and data types.
"""

import logging
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.constants import LOG_FORMAT, USD_TO_IDR_RATE

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

# --- Cleaning Functions ---


def clean_price(price_str: Optional[str]) -> Optional[float]:
    """Cleans the price string (USD) and converts it to float."""
    if price_str is None or "unavailable" in price_str.lower():
        return None
    try:
        # Remove '$', ',', and whitespace, then convert to float
        cleaned_price = re.sub(r"[$,]", "", price_str).strip()
        return float(cleaned_price)
    except (ValueError, TypeError):
        logging.warning("Could not parse price: '%s'. Returning None.", price_str)
        return None


def clean_rating(rating_str: Optional[str]) -> Optional[float]:
    """Cleans the rating string and extracts the numeric rating as float."""
    if (
        rating_str is None
        or "invalid" in rating_str.lower()
        or "not rated" in rating_str.lower()
    ):
        return None
    try:
        # Match patterns like '4.5 / 5' or '4.5' or '4 / 5' or '4'
        match = re.search(r"(\d(\.\d)?)\s*(?:/|\s|$)", rating_str)
        if match:
            return float(match.group(1))
        # Fallback: try converting directly if it's just a number string
        return float(rating_str.strip())
    except (ValueError, TypeError):
        logging.warning("Could not parse rating: '%s'. Returning None.", rating_str)
        return None


def clean_colors(colors_str: Optional[str]) -> Optional[int]:
    """Cleans the colors string and extracts the number of colors as int."""
    if colors_str is None:
        return None
    try:
        # Extract the first sequence of digits found
        match = re.search(r"(\d+)", colors_str)
        if match:
            return int(match.group(1))
        # Log if 'color' text is present but no number found
        if "color" in colors_str.lower():
            logging.warning(
                "Found 'color' text but no number in: '%s'. Returning None.", colors_str
            )
        return None
    except (ValueError, TypeError):
        logging.warning("Could not parse colors: '%s'. Returning None.", colors_str)
        return None


# --- Main Transformation Function ---


def transform_data(extracted_data: List[Dict[str, Optional[str]]]) -> pd.DataFrame:
    """
    Transforms extracted product data into a cleaned and structured DataFrame.

    Applies cleaning, filtering, currency conversion, duplicate removal,
    renaming, and type enforcement.

    Args:
        extracted_data: List of product dictionaries from the extraction step.
                        Expected keys: 'title', 'price', 'rating', 'colors',
                        'size', 'gender', 'image_url'.

    Returns:
        A cleaned, transformed Pandas DataFrame. Returns an empty DataFrame
        if input is empty or if all data is filtered out.
    """
    if not extracted_data:
        logging.warning(
            "Received empty list for transformation. Returning empty DataFrame."
        )
        return pd.DataFrame()

    try:
        df = pd.DataFrame(extracted_data)
        initial_rows = len(df)
        logging.info("Initial rows received for transformation: %d", initial_rows)
        if initial_rows == 0:
            return df  # Already empty

        # --- Step 1: Initial Cleaning and Preparation ---
        df["title"] = df["title"].str.strip()
        df["size"] = df["size"].str.strip()
        df["gender"] = df["gender"].str.strip()
        df["image_url"] = df["image_url"].str.strip()

        df["cleaned_price_usd"] = df["price"].apply(clean_price)
        df["cleaned_rating"] = df["rating"].apply(clean_rating)
        df["cleaned_colors"] = df["colors"].apply(clean_colors)

        # Add timestamp (convert UTC now to Asia/Jakarta)
        try:
            df["timestamp"] = pd.Timestamp.now(tz="UTC").tz_convert("Asia/Jakarta")
        except Exception as tz_error:
            logging.error(
                "Failed to convert timezone to Asia/Jakarta: %s. Using UTC.", tz_error
            )
            df["timestamp"] = pd.Timestamp.now(tz="UTC")  # Fallback to UTC

        # --- Step 2: Filter Invalid Data ("Unknown Product") ---
        unknown_mask = df["title"].str.contains("Unknown Product", na=False, case=False)
        df = df[~unknown_mask].copy()  # Use .copy() to avoid SettingWithCopyWarning
        rows_after_unknown = len(df)
        logging.info(
            "Rows after removing 'Unknown Product': %d (%d removed)",
            rows_after_unknown,
            initial_rows - rows_after_unknown,
        )
        if df.empty:
            logging.warning("DataFrame empty after filtering 'Unknown Product'.")
            return df

        # --- Step 3: Convert Price to IDR ---
        df["price_idr"] = df["cleaned_price_usd"] * USD_TO_IDR_RATE
        # Optional: Round IDR price to nearest integer or 2 decimal places
        # df['price_idr'] = df['price_idr'].round(0)

        # --- Step 4: Select and Prepare Final Columns ---
        # Columns needed for the final output structure BEFORE renaming
        final_columns_pre_rename = [
            "title",
            "price_idr",
            "cleaned_rating",
            "cleaned_colors",
            "size",
            "gender",
            "image_url",
            "timestamp",
        ]
        # Ensure all expected columns exist before selection
        missing_cols = [
            col for col in final_columns_pre_rename if col not in df.columns
        ]
        if missing_cols:
            logging.error(
                "Missing expected columns before final selection: %s", missing_cols
            )
            # Decide handling: raise error, return empty, or proceed with available?
            # For now, return empty to signal a schema mismatch issue.
            return pd.DataFrame()

        df_selected = df[final_columns_pre_rename].copy()

        # --- Step 5: Remove Rows with Null Values in Key Fields ---
        # Define which columns *must not* be null in the final output
        # Based on requirements, all final columns seem mandatory
        required_columns = df_selected.columns.tolist()
        rows_before_na = len(df_selected)
        df_selected.dropna(subset=required_columns, inplace=True)
        rows_after_na = len(df_selected)
        logging.info(
            "Rows after removing rows with ANY null in required columns (%s): %d (%d removed)",
            required_columns,
            rows_after_na,
            rows_before_na - rows_after_na,
        )
        if df_selected.empty:
            logging.warning("DataFrame empty after removing rows with null values.")
            return df_selected

        # --- Step 6: Remove Duplicate Rows ---
        rows_before_duplicates = len(df_selected)
        df_selected.drop_duplicates(inplace=True, keep="first")
        rows_after_duplicates = len(df_selected)
        logging.info(
            "Rows after removing duplicate rows: %d (%d removed)",
            rows_after_duplicates,
            rows_before_duplicates - rows_after_duplicates,
        )

        # --- Step 7: Rename Columns to Final Schema ---
        column_mapping = {
            "price_idr": "price",
            "cleaned_rating": "rating",
            "cleaned_colors": "colors",
            # 'title', 'size', 'gender', 'image_url', 'timestamp' remain the same
        }
        df_final = df_selected.rename(columns=column_mapping)
        logging.info("Columns renamed to final schema: %s", list(df_final.columns))

        # --- Step 8: Enforce Final Data Types ---
        try:
            type_mapping = {
                "title": str,
                "price": float,  # IDR Price
                "rating": float,
                "colors": int,
                "size": str,
                "gender": str,
                "image_url": str,
                # 'timestamp' should already be datetime64[ns, Asia/Jakarta] or datetime64[ns, UTC]
            }
            # Check if timestamp column exists before trying to access its dtype
            if "timestamp" in df_final.columns:
                # Ensure timestamp is not converted if already correct type
                if not pd.api.types.is_datetime64_any_dtype(df_final["timestamp"]):
                    # If it's somehow not a datetime, try converting (though unlikely)
                    df_final["timestamp"] = pd.to_datetime(df_final["timestamp"])
                # No explicit type needed in mapping if already correct

            # Apply type conversions for other columns
            df_final = df_final.astype(
                {k: v for k, v in type_mapping.items() if k in df_final.columns}
            )

            logging.info("Final data types enforced.")
            logging.debug("Final DataFrame head:\n%s", df_final.head().to_string())
            logging.debug("Final data types:\n%s", df_final.dtypes)

        except Exception as e:
            logging.error(
                "Error during final data type conversion: %s", e, exc_info=True
            )
            # Return the DataFrame before type casting attempt, as data is mostly processed
            return df_final

        final_rows = len(df_final)
        logging.info("Transformation complete. Final rows: %d", final_rows)
        logging.info(
            "Total rows removed during transformation: %d", initial_rows - final_rows
        )

        return df_final

    except KeyError as e:
        logging.error(
            "Missing expected key during transformation: %s. "
            "Check extraction keys and column names.",
            e,
            exc_info=True,
        )
        return pd.DataFrame()  # Return empty DataFrame on critical error
    except Exception as e:
        logging.error(
            "An unexpected error occurred during data transformation: %s",
            e,
            exc_info=True,
        )
        return pd.DataFrame()  # Return empty DataFrame on critical error


# --- Example Usage ---
if __name__ == "__main__":
    # Sample data mimicking extract output
    sample_extracted_data = [
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
        {
            "title": " T-shirt 1 ",
            "price": "$10.00",
            "rating": "⭐ 4.5 / 5",
            "colors": "3 colors",
            "size": " M ",
            "gender": " Men",
            "image_url": "url1",
        },  # Duplicate
        {
            "title": "Jacket 3",
            "price": "Price Unavailable",
            "rating": "⭐ 4.0 / 5",
            "colors": "2 colors",
            "size": "S",
            "gender": "Unisex",
            "image_url": "url3",
        },  # Null price
        {
            "title": "Unknown Product",
            "price": "$50.00",
            "rating": "⭐ 5.0 / 5",
            "colors": "1 Color",
            "size": "XL",
            "gender": "Men",
            "image_url": "url4",
        },  # Invalid name
        {
            "title": "Shoes 5",
            "price": "$120.00",
            "rating": "Not Rated",
            "colors": "4 colors",
            "size": "M",
            "gender": "Women",
            "image_url": "url5",
        },  # Null rating
        {
            "title": "Hat 6",
            "price": "$15.00",
            "rating": "⭐ 4.2 / 5",
            "colors": None,
            "size": "OS",
            "gender": "Unisex",
            "image_url": "url6",
        },  # Null colors
        {
            "title": "Belt 7",
            "price": "$30.00",
            "rating": "⭐ 4.9 / 5",
            "colors": "1 color",
            "size": None,
            "gender": "Unisex",
            "image_url": "url7",
        },  # Null size
        {
            "title": "Complete Item 8",
            "price": "$75.00",
            "rating": "⭐ 4.1 / 5",
            "colors": "2 colors",
            "size": "L",
            "gender": "Men",
            "image_url": "url8",
        },  # Valid row
        {
            "title": "Item 9 No Img",
            "price": "$40.00",
            "rating": "⭐ 4.0 / 5",
            "colors": "1 color",
            "size": "M",
            "gender": "Women",
            "image_url": None,
        },  # Null image_url
    ]
    print("\n--- Transformation Example ---")
    transformed_df = transform_data(sample_extracted_data)

    if not transformed_df.empty:
        print("\n--- Transformation Output ---")
        print(transformed_df.to_string())
        print("\nFinal Data types:")
        print(transformed_df.dtypes)
    else:
        print(
            "\nTransformation resulted in an empty DataFrame (all rows filtered or error)."
        )
