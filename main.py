# -*- coding: utf-8 -*-
"""
Main script to run the ETL pipeline for scraping fashion product data.

Orchestrates the extraction, transformation, and loading steps.
Configuration is loaded from environment variables managed via a .env file.
"""

import logging
import os
import sys
from typing import Dict, Any, Optional

import pandas as pd
from dotenv import load_dotenv

# Assuming utils package is structured correctly relative to this script
# If running as a package, these imports should work directly.
# If running as a script, ensure PYTHONPATH includes the project root.
from utils.constants import (
    POSTGRES_TABLE_NAME as DEFAULT_POSTGRES_TABLE,
    WORKSHEET_NAME as DEFAULT_WORKSHEET_NAME,
    LOG_FORMAT,
    CSV_FILEPATH as DEFAULT_CSV_FILEPATH,
)
from utils.extract import scrape_all_pages
from utils.load import load_to_csv, load_to_gsheets, load_to_postgres
from utils.transform import transform_data

# Configure logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def load_configuration() -> Dict[str, Any]:
    """
    Loads configuration from environment variables.

    Returns:
        A dictionary containing the configuration settings.

    Raises:
        ValueError: If essential configuration like SOURCE_URL is missing.
    """
    load_dotenv()  # Load .env file if present

    config: Dict[str, Any] = {}

    # --- Essential Configuration ---
    config["source_url"] = os.getenv("SOURCE_URL")
    if not config["source_url"]:
        logging.error("SOURCE_URL environment variable not set. Cannot proceed.")
        raise ValueError("Missing SOURCE_URL configuration.")

    # --- Extraction Configuration ---
    try:
        # Use a sensible default if not set or invalid
        config["max_pages"] = int(os.getenv("MAX_PAGES_TO_SCRAPE", "50"))
    except ValueError:
        logging.warning(
            "Invalid MAX_PAGES_TO_SCRAPE value. Using default: 50"
        )
        config["max_pages"] = 50

    # --- Load Target Configuration ---

    # CSV
    config["csv_filepath"] = os.getenv("CSV_FILEPATH", DEFAULT_CSV_FILEPATH)

    # Google Sheets
    gsheets_credentials_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")
    gsheets_sheet_id = os.getenv("GOOGLE_SHEET_ID")
    config["gsheets"] = {
        "credentials_path": gsheets_credentials_path,
        "sheet_id": gsheets_sheet_id,
        "worksheet_name": os.getenv("WORKSHEET_NAME", DEFAULT_WORKSHEET_NAME),
        "enabled": bool(gsheets_credentials_path and gsheets_sheet_id),
    }

    # PostgreSQL
    db_config = {
        "host": os.getenv("DB_HOST"),
        # Use string default for port as getenv expects str or None
        "port": os.getenv("DB_PORT", "5432"),
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }
    postgres_config_complete = all(
        db_config.get(k) for k in ["host", "dbname", "user", "password"]
    )
    config["postgres"] = {
        "db_config": db_config,
        "table_name": os.getenv("POSTGRES_TABLE_NAME", DEFAULT_POSTGRES_TABLE),
        "enabled": postgres_config_complete,
    }

    return config


def _run_extraction(config: Dict[str, Any]) -> Optional[list]:
    """Runs the data extraction step."""
    logging.info(
        "Step 1: Extracting data from %s (max %d pages)...",
        config["source_url"],
        config["max_pages"],
    )
    try:
        all_extracted_data = scrape_all_pages(
            config["source_url"], config["max_pages"]
        )
        if not all_extracted_data:
            logging.warning("Extraction yielded no data.")
            return None
        logging.info(
            "Extraction complete. Total raw items extracted: %d",
            len(all_extracted_data),
        )
        return all_extracted_data
    except Exception as e:  # pylint: disable=broad-except
        logging.error("Extraction failed: %s", e, exc_info=True)
        return None


def _run_transformation(extracted_data: list) -> Optional[pd.DataFrame]:
    """Runs the data transformation step."""
    logging.info("Step 2: Transforming extracted data...")
    try:
        transformed_df = transform_data(extracted_data)
        if transformed_df is None:
            # transform_data should ideally return empty DataFrame, not None
            logging.error("Transformation step returned None unexpectedly.")
            return None
        if transformed_df.empty:
            logging.warning(
                "Transformation resulted in an empty DataFrame. "
                "No data will be loaded."
            )
            return None  # Return None to signal no data to load

        logging.info(
            "Transformation complete. Processed %d valid products.",
            len(transformed_df),
        )
        logging.debug(
            "Sample of transformed data:\n%s", transformed_df.head().to_string()
        )
        return transformed_df
    except Exception as e:  # pylint: disable=broad-except
        logging.error("Transformation failed: %s", e, exc_info=True)
        return None


def _load_data_to_csv(
    dataframe: pd.DataFrame, config: Dict[str, Any]
) -> None:
    """Loads data to a CSV file."""
    filepath = config["csv_filepath"]
    logging.info("Attempting to load data to CSV file: %s", filepath)
    try:
        if load_to_csv(dataframe, filepath):
            logging.info("Successfully loaded data to %s", filepath)
        else:
            # load_to_csv returning False indicates a handled failure
            logging.error("Failed to load data to %s", filepath)
    except Exception as e:  # pylint: disable=broad-except
        logging.error(
            "Unhandled exception during CSV load to %s: %s", filepath, e, exc_info=True
        )


def _load_data_to_gsheets(
    dataframe: pd.DataFrame, config: Dict[str, Any]
) -> None:
    """Loads data to Google Sheets if configured."""
    gsheets_config = config["gsheets"]
    if not gsheets_config["enabled"]:
        logging.info(
            "Skipping Google Sheets load: GOOGLE_SHEET_ID or "
            "GOOGLE_SHEETS_CREDENTIALS_PATH not configured."
        )
        return

    credentials_path = gsheets_config["credentials_path"]
    sheet_id = gsheets_config["sheet_id"]
    worksheet_name = gsheets_config["worksheet_name"]

    if not os.path.exists(credentials_path):
        logging.warning(
            "Skipping Google Sheets load: Credentials file not found at %s",
            credentials_path,
        )
        return

    logging.info(
        "Attempting to load data to Google Sheet ID: %s, Worksheet: %s",
        sheet_id,
        worksheet_name,
    )
    try:
        if load_to_gsheets(
            dataframe, credentials_path, sheet_id, worksheet_name
        ):
            logging.info("Successfully loaded data to Google Sheets.")
        else:
            logging.error("Failed to load data to Google Sheets.")
    except Exception as e:  # pylint: disable=broad-except
        logging.error(
            "Unhandled exception during Google Sheets load: %s", e, exc_info=True
        )


def _load_data_to_postgres(
    dataframe: pd.DataFrame, config: Dict[str, Any]
) -> None:
    """Loads data to PostgreSQL if configured."""
    postgres_config = config["postgres"]
    if not postgres_config["enabled"]:
        logging.info(
            "Skipping PostgreSQL load: Database configuration is incomplete."
        )
        return

    db_conn_info = postgres_config["db_config"]
    table_name = postgres_config["table_name"]

    logging.info(
        "Attempting to load data to PostgreSQL table: %s.%s",
        db_conn_info.get("dbname"),
        table_name,
    )
    try:
        if load_to_postgres(dataframe, db_conn_info, table_name):
            logging.info("Successfully loaded data to PostgreSQL.")
        else:
            logging.error("Failed to load data to PostgreSQL.")
    except Exception as e:  # pylint: disable=broad-except
        logging.error(
            "Unhandled exception during PostgreSQL load: %s", e, exc_info=True
        )


def run_pipeline(config: Dict[str, Any]) -> None:
    """
    Executes the full ETL pipeline: Extract, Transform, Load.

    Args:
        config: Dictionary containing pipeline configuration.
    """
    logging.info("========================================")
    logging.info("Starting ETL pipeline...")
    logging.info("========================================")

    # --- Extract ---
    extracted_data = _run_extraction(config)
    if not extracted_data:
        logging.warning("Extraction step failed or yielded no data. Pipeline terminating.")
        return

    # --- Transform ---
    transformed_df = _run_transformation(extracted_data)
    if transformed_df is None or transformed_df.empty:
        logging.warning(
            "Transformation step failed or resulted in empty data. "
            "No data will be loaded."
        )
        return

    # --- Load ---
    logging.info("Step 3: Loading transformed data...")
    _load_data_to_csv(transformed_df, config)
    _load_data_to_gsheets(transformed_df, config)
    _load_data_to_postgres(transformed_df, config)

    logging.info("========================================")
    logging.info("ETL pipeline finished.")
    logging.info("========================================")


def main() -> None:
    """Main function to load configuration and run the pipeline."""
    try:
        configuration = load_configuration()
        run_pipeline(configuration)
    except ValueError as e:
        logging.error("Configuration error: %s", e)
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-except
        logging.critical("An unexpected error occurred: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
    