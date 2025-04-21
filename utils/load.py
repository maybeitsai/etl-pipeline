# utils/load.py
"""
Module for loading the transformed DataFrame into various destinations:
CSV files, Google Sheets, and PostgreSQL database.
"""

import logging
import os
from typing import Dict

import gspread
import numpy as np
import pandas as pd
import psycopg2
from google.oauth2.service_account import Credentials
from psycopg2 import sql
from psycopg2.extras import execute_values

from utils.constants import WORKSHEET_NAME, LOG_FORMAT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


# --- Load to CSV ---
def load_to_csv(df: pd.DataFrame, filepath: str) -> bool:
    """
    Saves the DataFrame to a CSV file, overwriting if it exists.

    Args:
        df: The DataFrame to save.
        filepath: The path to the output CSV file.

    Returns:
        True if successful, False otherwise.
    """
    if df.empty:
        logging.warning("DataFrame is empty. Skipping CSV save to %s.", filepath)
        # Return True as there's no error, just nothing to save.
        # Could also return False if an empty CSV is undesirable. Let's return True.
        return True
    try:
        # Ensure directory exists if filepath includes directories
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        df.to_csv(filepath, index=False, encoding="utf-8")
        logging.info("Successfully saved %d rows to CSV: %s", len(df), filepath)
        return True
    except IOError as e:
        logging.error("Error saving data to CSV file %s: %s", filepath, e)
        return False
    except Exception as e:
        logging.error(
            "An unexpected error occurred during CSV saving: %s", e, exc_info=True
        )
        return False


# --- Load to Google Sheets ---
def load_to_gsheets(
    df: pd.DataFrame,
    credentials_path: str,
    sheet_id: str,
    worksheet_name: str = WORKSHEET_NAME,
) -> bool:
    """
    Loads the DataFrame into a Google Sheet, clearing existing data first.

    Args:
        df: The DataFrame to load.
        credentials_path: Path to the Google Service Account JSON key file.
        sheet_id: The ID of the Google Sheet.
        worksheet_name: The name of the worksheet to load data into.

    Returns:
        True if successful, False otherwise.
    """
    if df.empty:
        logging.warning(
            "DataFrame is empty. Skipping Google Sheets load to sheet ID %s, worksheet '%s'.",
            sheet_id,
            worksheet_name,
        )
        return True  # No error, just nothing to load.

    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(credentials_path, scopes=scopes)
        gc = gspread.authorize(creds)
        spreadsheet = gc.open_by_key(sheet_id)

        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            logging.info("Worksheet '%s' not found. Creating it.", worksheet_name)
            # Create with sufficient rows/cols initially? Or let gspread handle it?
            # Let gspread handle resizing for simplicity.
            worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=1, cols=1)

        # Prepare data for gspread:
        # 1. Convert NaN/NaT to None (gspread handles None as empty cell)
        # 2. Convert Timestamp to ISO 8601 string format (safer for Sheets)
        df_gspread = df.copy()
        if "timestamp" in df_gspread.columns and pd.api.types.is_datetime64_any_dtype(
            df_gspread["timestamp"]
        ):
            # Ensure timezone info is preserved if present
            df_gspread["timestamp"] = df_gspread["timestamp"].dt.strftime(
                "%Y-%m-%d %H:%M:%S%z"
            )

        # Replace remaining NaN/NaT with None, then convert all to list of lists
        # gspread prefers None for empty cells over empty strings usually.
        df_gspread = df_gspread.replace({np.nan: None, pd.NaT: None})
        data_to_load = [df_gspread.columns.values.tolist()] + df_gspread.values.tolist()

        # Clear existing content and update
        worksheet.clear()
        # Update the sheet with headers and data
        worksheet.update(
            data_to_load,
            value_input_option="USER_ENTERED",  # Interprets values as if typed by user
        )

        logging.info(
            "Successfully loaded %d rows to Google Sheet ID: %s, Worksheet: %s",
            len(df),
            sheet_id,
            worksheet_name,
        )
        return True

    except gspread.exceptions.APIError as e:
        logging.error("Google Sheets API error: %s", e)
        return False
    except FileNotFoundError:
        logging.error("Credentials file not found at: %s", credentials_path)
        return False
    except Exception as e:
        logging.error(
            "An unexpected error occurred during Google Sheets loading: %s",
            e,
            exc_info=True,
        )
        return False


# --- Load to PostgreSQL ---
def _get_postgres_schema(df: pd.DataFrame) -> str:
    """Generates a basic CREATE TABLE statement schema based on DataFrame dtypes."""
    # Mapping from Pandas dtype to PostgreSQL type
    # This is a basic mapping and might need refinement based on specific data needs
    type_map = {
        "object": "TEXT",  # For strings, URLs, etc.
        "int64": "INTEGER",
        "float64": "REAL",  # Or NUMERIC(precision, scale) for exact decimals like price
        "bool": "BOOLEAN",
        "datetime64[ns]": "TIMESTAMP",  # Use TIMESTAMP WITHOUT TIME ZONE
        # Handle timezone-aware datetime specifically
        # Check for specific timezone dtype string representation if needed
    }

    column_definitions = []
    for col_name, dtype in df.dtypes.items():
        # Handle timezone-aware timestamp specifically
        if (
            pd.api.types.is_datetime64_any_dtype(dtype)
            and getattr(dtype, "tz", None) is not None
        ):
            pg_type = "TIMESTAMPTZ"  # Use TIMESTAMP WITH TIME ZONE
        elif dtype.name in type_map:
            pg_type = type_map[dtype.name]
            # Refine float to NUMERIC for price column if desired
            if col_name == "price" and pg_type == "REAL":
                pg_type = "NUMERIC(12, 2)"  # Example: 12 total digits, 2 decimal places
        else:
            logging.warning(
                "Unmapped dtype '%s' for column '%s'. Defaulting to TEXT.",
                dtype.name,
                col_name,
            )
            pg_type = "TEXT"

        # Use sql.Identifier for safe column naming
        column_definitions.append(
            sql.SQL("{} {}").format(sql.Identifier(col_name), sql.SQL(pg_type))
        )

    # Combine column definitions into the CREATE TABLE statement
    schema_sql = sql.SQL(", ").join(column_definitions)
    return schema_sql


def load_to_postgres(
    df: pd.DataFrame, db_config: Dict[str, str], table_name: str
) -> bool:
    """
    Loads the DataFrame into a PostgreSQL table.
    Creates table based on DataFrame schema if not exists, then replaces data (TRUNCATE + INSERT).

    Args:
        df: The DataFrame to load.
        db_config: Dictionary with DB connection details
                   (host, port, dbname, user, password).
        table_name: The name of the target table in PostgreSQL.

    Returns:
        True if successful, False otherwise.
    """
    if df.empty:
        logging.warning(
            "DataFrame is empty. Skipping PostgreSQL load to table '%s'.", table_name
        )
        return True  # No error, just nothing to load.

    conn = None
    try:
        conn = psycopg2.connect(**db_config)
        conn.autocommit = False  # Ensure operations are transactional
        with conn.cursor() as cursor:
            logging.info(
                "Successfully connected to PostgreSQL database '%s'",
                db_config.get("dbname"),
            )

            # Generate schema based on DataFrame
            schema_sql = _get_postgres_schema(df)
            create_table_query = sql.SQL(
                "CREATE TABLE IF NOT EXISTS {table} ({schema});"
            ).format(table=sql.Identifier(table_name), schema=schema_sql)

            cursor.execute(create_table_query)
            logging.info("Ensured table '%s' exists.", table_name)

            # Clear existing data (TRUNCATE is usually faster than DELETE)
            truncate_query = sql.SQL("TRUNCATE TABLE {table};").format(
                table=sql.Identifier(table_name)
            )
            cursor.execute(truncate_query)
            logging.info("Truncated table '%s' before loading new data.", table_name)

            # Prepare data for insertion (handle NaN/NaT -> None for SQL compatibility)
            data_tuples = [
                tuple(x) for x in df.replace({np.nan: None, pd.NaT: None}).to_numpy()
            ]
            if not data_tuples:
                logging.info("No valid data tuples to insert into PostgreSQL.")
                conn.commit()  # Commit transaction even if no data inserted (table created/truncated)
                return True

            # Prepare INSERT statement using execute_values for efficiency
            cols = sql.SQL(", ").join(map(sql.Identifier, df.columns))
            insert_query = sql.SQL("INSERT INTO {table} ({cols}) VALUES %s").format(
                table=sql.Identifier(table_name), cols=cols
            )

            # Execute bulk insert
            execute_values(cursor, insert_query.as_string(conn), data_tuples)
            logging.info(
                "Attempting to load %d records into PostgreSQL table '%s'.",
                len(data_tuples),
                table_name,
            )

        # Commit the transaction if all steps succeeded
        conn.commit()
        logging.info(
            "Successfully loaded %d records into PostgreSQL table '%s'. Transaction committed.",
            len(data_tuples),
            table_name,
        )
        return True

    except psycopg2.Error as e:
        logging.error("PostgreSQL error during load to table '%s': %s", table_name, e)
        if conn:
            conn.rollback()  # Rollback transaction on error
            logging.info("PostgreSQL transaction rolled back.")
        return False
    except Exception as e:
        logging.error(
            "An unexpected error occurred during PostgreSQL loading to table '%s': %s",
            table_name,
            e,
            exc_info=True,
        )
        if conn:
            conn.rollback()
            logging.info("PostgreSQL transaction rolled back due to unexpected error.")
        return False
    finally:
        if conn:
            conn.close()
            logging.info("PostgreSQL connection closed.")


# Example Usage (Optional - requires environment setup)
if __name__ == "__main__":
    # Create Sample DataFrame (ensure dtypes match expected final transform output)
    sample_data = {
        "title": ["Test Product 1", "Test Product 2"],
        "price": [160000.00, 2392000.00],  # Example IDR float
        "rating": [4.5, 3.8],
        "colors": [3, 5],
        "size": ["M", "L"],
        "gender": ["Unisex", "Men"],
        "image_url": ["url1", "url2"],
        "timestamp": [
            pd.Timestamp.now(tz="Asia/Jakarta"),
            pd.Timestamp.now(tz="Asia/Jakarta"),
        ],
    }
    sample_df = pd.DataFrame(sample_data)
    # Enforce types similar to transform output
    sample_df = sample_df.astype(
        {
            "title": str,
            "price": float,
            "rating": float,
            "colors": int,
            "size": str,
            "gender": str,
            "image_url": str,
        }
    )
    # Timestamp is already correct type

    # --- Test CSV ---
    CSV_FILE = "test_products_output.csv"
    print(f"\n--- Testing CSV Load to {CSV_FILE} ---")
    load_to_csv(sample_df, CSV_FILE)

    # --- Test Google Sheets (Requires setup) ---
    print(
        "\n--- Testing Google Sheets Load (requires credentials and sheet ID in .env) ---"
    )
    CREDS_FILE = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")
    SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
    WORKSHEET = os.getenv("WORKSHEET_NAME", WORKSHEET_NAME)

    if CREDS_FILE and os.path.exists(CREDS_FILE) and SHEET_ID:
        load_to_gsheets(sample_df, CREDS_FILE, SHEET_ID, WORKSHEET)
    else:
        print(
            "Skipping Google Sheets test: Credentials path or Sheet ID not found/configured in .env"
        )

    # --- Test PostgreSQL (Requires running DB and .env config) ---
    print("\n--- Testing PostgreSQL Load (requires running DB and .env config) ---")
    DB_PARAMS = {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }
    TABLE_NAME = "test_load_products"

    # Basic check if essential DB config seems present
    if all(DB_PARAMS.get(k) for k in ["host", "dbname", "user", "password"]):
        load_to_postgres(sample_df, DB_PARAMS, table_name=TABLE_NAME)
    else:
        print("Skipping PostgreSQL test: DB configuration incomplete in .env")
