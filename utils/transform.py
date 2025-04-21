# utils/transform.py
import pandas as pd
import numpy as np
import re
import logging
from typing import List, Dict, Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def clean_price(price_str: Optional[str]) -> Optional[float]:
    """Cleans the price string and converts it to float."""
    if price_str is None or price_str == "Price Unavailable":
        return None
    try:
        # Remove '$', commas, and whitespace, then convert to float
        cleaned_price = re.sub(r"[$,]", "", price_str).strip()
        return float(cleaned_price)
    except (ValueError, TypeError):
        logging.warning(f"Could not parse price: {price_str}")
        return None


def clean_rating(rating_str: Optional[str]) -> Optional[float]:
    """Cleans the rating string and extracts the numeric rating."""
    if rating_str is None or rating_str in ["Invalid Rating / 5", "Not Rated"]:
        return None
    try:
        # Extract the numeric part (e.g., '3.9' from '⭐ 3.9 / 5')
        match = re.search(r"(\d\.\d)", rating_str)
        if match:
            return float(match.group(1))
        else:
            # Handle cases where only the number might be present (less likely based on HTML)
            return float(rating_str)
    except (ValueError, TypeError):
        logging.warning(f"Could not parse rating: {rating_str}")
        return None


def clean_colors(colors_str: Optional[str]) -> Optional[int]:
    """Cleans the colors string and extracts the number of colors."""
    if colors_str is None:
        return None
    try:
        # Extract the number (e.g., '3' from '3 Colors')
        match = re.search(r"(\d+)", colors_str)
        if match:
            return int(match.group(1))
        else:
            return None  # Or handle differently if format varies
    except (ValueError, TypeError):
        logging.warning(f"Could not parse colors: {colors_str}")
        return None


def transform_data(
    extracted_data: List[Dict[str, Optional[str]]],
) -> Optional[pd.DataFrame]:
    """
    Transforms the extracted product data list into a cleaned Pandas DataFrame.

    Args:
        extracted_data (List[Dict[str, Optional[str]]]): List of product dictionaries
                                                          from the extraction step.

    Returns:
        Optional[pd.DataFrame]: A cleaned Pandas DataFrame, or None if input is invalid.
    """
    if not extracted_data:
        logging.warning("Received empty or invalid data for transformation.")
        return None

    try:
        df = pd.DataFrame(extracted_data)

        # Apply cleaning functions
        df["price"] = df["price"].apply(clean_price)
        df["rating"] = df["rating"].apply(clean_rating)
        df["num_colors"] = df["colors"].apply(clean_colors)

        # Clean text fields (strip whitespace)
        df["product_name"] = df["product_name"].str.strip()
        df["size"] = df["size"].str.strip()
        df["gender"] = df["gender"].str.strip()

        # Handle 'Unknown Product' - replace with NaN or keep as is? Let's keep for now.
        # df['product_name'] = df['product_name'].replace('Unknown Product', np.nan)

        # Add extraction timestamp
        df["extracted_at"] = pd.Timestamp.now(tz="UTC").tz_convert(
            "Asia/Jakarta"
        )  # Use UTC for consistency

        # Select and reorder columns for clarity
        final_columns = [
            "product_name",
            "price",
            "rating",
            "num_colors",
            "size",
            "gender",
            "image_url",
            "extracted_at",
            # Can drop original 'price', 'rating', 'colors' if desired
        ]
        df_transformed = df[final_columns]

        logging.info(f"Successfully transformed {len(df_transformed)} records.")
        return df_transformed

    except KeyError as e:
        logging.error(
            f"Missing expected key during transformation: {e}. Input data might be malformed."
        )
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during data transformation: {e}")
        return None


# # Example usage (optional)
# if __name__ == '__main__':
#     # Sample data mimicking extract output
#     sample_extracted_data = [
#         {'product_name': 'T-shirt 2', 'price': '$102.15', 'rating': '⭐ 3.9 / 5', 'colors': '3 Colors', 'size': 'M', 'gender': 'Women', 'image_url': 'url1'},
#         {'product_name': 'Pants 16', 'price': 'Price Unavailable', 'rating': 'Not Rated', 'colors': '8 Colors', 'size': 'S', 'gender': 'Men', 'image_url': 'url2'},
#         {'product_name': 'Unknown Product', 'price': '$100.00', 'rating': '⭐ Invalid Rating / 5', 'colors': '5 Colors', 'size': 'M', 'gender': 'Men', 'image_url': 'url3'}
#     ]
#     transformed_df = transform_data(sample_extracted_data)
#     if transformed_df is not None:
#         print("Transformation successful:")
#         print(transformed_df.head())
#         print("\nData types:")
#         print(transformed_df.dtypes)
#     else:
#         print("Transformation failed.")
