# utils/load.py
"""
Module for loading the transformed DataFrame into various destinations:
CSV files, Google Sheets, and PostgreSQL database.
"""

import logging
import os
from typing import Dict

import gspread
import pandas as pd
import psycopg2
from google.oauth2.service_account import Credentials
from gspread.exceptions import GSpreadException
from psycopg2 import sql
from psycopg2.extras import execute_values
from psycopg2 import Error as Psycopg2Error

from utils.constants import LOG_FORMAT, WORKSHEET_NAME

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
        return True  # No error, just nothing to save.

    try:
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        df.to_csv(filepath, index=False, encoding="utf-8")
        logging.info("Successfully saved %d rows to CSV: %s", len(df), filepath)
        return True
    # Catch specific file-related errors
    except (IOError, OSError) as e:
        logging.error("Error saving data to CSV file %s: %s", filepath, e)
        return False
    # Catch unexpected errors during CSV writing (e.g., pandas issues)
    except Exception as e:  # pylint: disable=broad-except
        logging.error(
            "An unexpected error occurred during CSV saving to %s: %s",
            filepath,
            e,
            exc_info=True,
        )
        return False


# --- Load to Google Sheets ---


def _prepare_gsheets_data(df: pd.DataFrame) -> list:
    """Prepares DataFrame data for gspread upload."""
    df_gspread = df.copy()
    # Handle timestamp conversion to string
    if "timestamp" in df_gspread.columns and pd.api.types.is_datetime64_any_dtype(
        df_gspread["timestamp"]
    ):
        df_gspread["timestamp"] = df_gspread["timestamp"].dt.strftime(
            "%Y-%m-%d %H:%M:%S%z"
        )

    # Replace NaN/NaT with None (gspread handles None as empty cell)
    # Convert all data to object type before replacing to avoid potential issues
    # with pandas internal types, then convert to list of lists.
    df_gspread = df_gspread.astype(object).where(pd.notnull(df_gspread), None)
    return [df_gspread.columns.values.tolist()] + df_gspread.values.tolist()


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
            worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=1, cols=1)

        data_to_load = _prepare_gsheets_data(df)

        # Clear existing content and update
        worksheet.clear()
        worksheet.update(data_to_load, value_input_option="USER_ENTERED")

        logging.info(
            "Successfully loaded %d rows to Google Sheet ID: %s, Worksheet: %s",
            len(df),
            sheet_id,
            worksheet_name,
        )
        return True

    # Catch specific gspread exceptions
    except GSpreadException as e:
        logging.error("Google Sheets API or client error: %s", e)
        return False
    except FileNotFoundError:
        logging.error("Credentials file not found at: %s", credentials_path)
        return False
    except ValueError as e:
        logging.error("Value error during Google Sheets operation: %s", e)
        return False
    # Catch unexpected errors during Sheets interaction
    except Exception as e:  # pylint: disable=broad-except
        logging.error(
            "An unexpected error occurred during Google Sheets loading: %s",
            e,
            exc_info=True,
        )
        return False


# --- Load to PostgreSQL ---
def _get_postgres_schema(df: pd.DataFrame) -> sql.SQL:
    """Generates a basic CREATE TABLE schema based on DataFrame dtypes."""
    type_map = {
        "object": "TEXT",
        "int64": "INTEGER",
        "float64": "REAL",  # Consider NUMERIC for exact decimals
        "bool": "BOOLEAN",
        "datetime64[ns]": "TIMESTAMP",  # Timezone naive
    }

    column_definitions = []
    for col_name, dtype in df.dtypes.items():
        pg_type = None
        # Handle timezone-aware timestamp specifically
        if pd.api.types.is_datetime64_any_dtype(dtype) and getattr(dtype, "tz", None):
            pg_type = "TIMESTAMPTZ"  # Use TIMESTAMP WITH TIME ZONE
        elif dtype.name in type_map:
            pg_type = type_map[dtype.name]
            # Refine float to NUMERIC for price column if desired
            if col_name == "price" and pg_type == "REAL":
                pg_type = "NUMERIC(12, 2)"  # Example precision
        else:
            logging.warning(
                "Unmapped dtype '%s' for column '%s'. Defaulting to TEXT.",
                dtype.name,
                col_name,
            )
            pg_type = "TEXT"

        column_definitions.append(
            sql.SQL("{} {}").format(sql.Identifier(col_name), sql.SQL(pg_type))
        )

    return sql.SQL(", ").join(column_definitions)


def load_to_postgres(
    df: pd.DataFrame, db_config: Dict[str, str], table_name: str
) -> bool:
    """
    Loads the DataFrame into a PostgreSQL table. Creates/replaces table data.

    Args:
        df: The DataFrame to load.
        db_config: Dict with DB connection details (host, port, dbname, user, password).
        table_name: The name of the target table.

    Returns:
        True if successful, False otherwise.
    """
    if df.empty:
        logging.warning(
            "DataFrame is empty. Skipping PostgreSQL load to table '%s'.", table_name
        )
        return True

    conn = None
    try:
        conn = psycopg2.connect(**db_config)
        conn.autocommit = False  # Use transactions
        with conn.cursor() as cursor:
            logging.info(
                "Connected to PostgreSQL database '%s'", db_config.get("dbname")
            )

            # Generate schema and create table if not exists
            schema_sql = _get_postgres_schema(df)
            create_table_query = sql.SQL(
                "CREATE TABLE IF NOT EXISTS {table} ({schema});"
            ).format(table=sql.Identifier(table_name), schema=schema_sql)
            cursor.execute(create_table_query)
            logging.info("Ensured table '%s' exists.", table_name)

            # Clear existing data
            truncate_query = sql.SQL("TRUNCATE TABLE {table};").format(
                table=sql.Identifier(table_name)
            )
            cursor.execute(truncate_query)
            logging.info("Truncated table '%s'.", table_name)

            # Prepare data for insertion (handle NaN/NaT -> None)
            # Use astype(object) before replacing NaN to handle mixed types robustly
            df_prepared = df.astype(object).where(pd.notnull(df), None)
            data_tuples = [tuple(x) for x in df_prepared.to_numpy()]

            if not data_tuples:
                logging.info("No valid data tuples to insert into PostgreSQL.")
                conn.commit()  # Commit transaction (table created/truncated)
                return True

            # Prepare and execute bulk insert
            cols = sql.SQL(", ").join(map(sql.Identifier, df.columns))
            insert_query = sql.SQL("INSERT INTO {table} ({cols}) VALUES %s").format(
                table=sql.Identifier(table_name), cols=cols
            )
            execute_values(
                cursor, insert_query.as_string(cursor.connection), data_tuples
            )
            logging.info(
                "Attempting to load %d records into PostgreSQL table '%s'.",
                len(data_tuples),
                table_name,
            )

        conn.commit()  # Commit transaction
        logging.info(
            "Successfully loaded %d records into PostgreSQL table '%s'. Committed.",
            len(data_tuples),
            table_name,
        )
        return True

    # Catch specific database errors
    except Psycopg2Error as e:
        logging.error("PostgreSQL error during load to table '%s': %s", table_name, e)
        if conn:
            conn.rollback()
            logging.info("PostgreSQL transaction rolled back.")
        return False
    # Catch errors related to db_config dictionary
    except KeyError as e:
        logging.error("Missing key in db_config: %s", e)
        return False
    # Catch unexpected errors during DB operations
    # Catch unexpected errors during DB operations
    except Exception as e:  # pylint: disable=broad-except
        logging.error(
            "An unexpected error occurred during PostgreSQL load to '%s': %s. PostgreSQL transaction rolled back.",
            table_name,
            e,
            exc_info=True,
        )
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()
            logging.info("PostgreSQL connection closed.")


# Example Usage (Optional - requires environment setup)
# if __name__ == "__main__":
#     # Create Sample DataFrame
#     sample_data = {
#         "title": ["Test Product 1", "Test Product 2"],
#         "price": [160000.00, 2392000.00],
#         "rating": [4.5, 3.8],
#         "colors": [3, 5],
#         "size": ["M", "L"],
#         "gender": ["Unisex", "Men"],
#         "image_url": ["url1", "url2"],
#         "timestamp": [
#             pd.Timestamp.now(tz="Asia/Jakarta"),
#             pd.Timestamp.now(tz="Asia/Jakarta"),
#         ],
#     }
#     sample_df = pd.DataFrame(sample_data).astype(
#         {
#             "title": str,
#             "price": float,
#             "rating": float,
#             "colors": int,
#             "size": str,
#             "gender": str,
#             "image_url": str,
#         }
#     )

#     # --- Test CSV ---
#     CSV_TEST_FILE = "test_products_output.csv"
#     print(f"\n--- Testing CSV Load to {CSV_TEST_FILE} ---")
#     load_to_csv(sample_df, CSV_TEST_FILE)

#     # --- Test Google Sheets (Requires setup) ---
#     print("\n--- Testing Google Sheets Load (requires env vars) ---")
#     CREDS_FILE = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")
#     SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
#     WORKSHEET = os.getenv("WORKSHEET_NAME", WORKSHEET_NAME)

#     if CREDS_FILE and os.path.exists(CREDS_FILE) and SHEET_ID:
#         load_to_gsheets(sample_df, CREDS_FILE, SHEET_ID, WORKSHEET)
#     else:
#         print("Skipping GSheets test: Env vars GOOGLE SHEET CREDENTIALS/ID missing.")

#     # --- Test PostgreSQL (Requires setup) ---
#     print("\n--- Testing PostgreSQL Load (requires env vars) ---")
#     DB_PARAMS = {
#         "host": os.getenv("DB_HOST"),
#         "port": os.getenv("DB_PORT"),
#         "dbname": os.getenv("DB_NAME"),
#         "user": os.getenv("DB_USER"),
#         "password": os.getenv("DB_PASSWORD"),
#     }
#     TABLE_NAME = "test_load_products"

#     if all(DB_PARAMS.values()):  # Basic check if all DB params are set
#         load_to_postgres(sample_df, DB_PARAMS, table_name=TABLE_NAME)
#     else:
#         print("Skipping PostgreSQL test: DB config incomplete in env vars.")
