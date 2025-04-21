# utils/load.py
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import gspread
from google.oauth2.service_account import Credentials
import logging
import os
from typing import Dict, Optional
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Load to CSV ---
def load_to_csv(df: pd.DataFrame, filepath: str) -> bool:
    """
    Saves the DataFrame to a CSV file. Overwrites if exists.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filepath (str): The path to the output CSV file.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        df.to_csv(filepath, index=False, encoding="utf-8")
        logging.info(f"Successfully saved data to CSV: {filepath}")
        return True
    except IOError as e:
        logging.error(f"Error saving data to CSV file {filepath}: {e}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during CSV saving: {e}")
        return False


# --- Load to Google Sheets ---
def load_to_gsheets(
    df: pd.DataFrame,
    credentials_path: str,
    sheet_id: str,
    worksheet_name: str = "Sheet1",
) -> bool:
    """
    Loads the DataFrame into a Google Sheet. Clears existing data first.

    Args:
        df (pd.DataFrame): The DataFrame to load.
        credentials_path (str): Path to the Google Service Account JSON key file.
        sheet_id (str): The ID of the Google Sheet.
        worksheet_name (str): The name of the worksheet to load data into.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(credentials_path, scopes=scopes)
        gc = gspread.authorize(creds)

        spreadsheet = gc.open_by_key(sheet_id)

        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            logging.info(f"Worksheet '{worksheet_name}' not found. Creating it.")
            worksheet = spreadsheet.add_worksheet(
                title=worksheet_name, rows="1", cols="1"
            )

        # Buat salinan DataFrame untuk menghindari modifikasi objek asli
        df_copy = df.copy()

        # Konversi kolom Timestamp ke string format ISO 8601
        # Cek dulu apakah kolomnya ada
        if "extracted_at" in df_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(df_copy["extracted_at"]):
                df_copy["extracted_at"] = df_copy["extracted_at"].dt.strftime(
                    "%Y-%m-%d %H:%M:%S.%f"
                )
            else:
                df_copy["extracted_at"] = df_copy["extracted_at"].astype(str)

        # Siapkan data untuk gspread (konversi NaN/NaT/None ke string kosong '')
        # Google Sheets lebih baik menangani string kosong daripada None untuk sel kosong
        df_string = df_copy.astype(str)
        missing_value_strings = ["nan", "NaT", "<NA>", "None"]
        df_gspread = df_string.replace(missing_value_strings, "")

        # Clear existing content and update
        worksheet.clear()
        worksheet.update(
            [df_gspread.columns.values.tolist()] + df_gspread.values.tolist(),
            value_input_option="USER_ENTERED",
        )

        logging.info(
            f"Successfully loaded data to Google Sheet ID: {sheet_id}, Worksheet: {worksheet_name}"
        )
        return True

    except gspread.exceptions.APIError as e:
        logging.error(f"Google Sheets API error: {e}")
        return False
    except FileNotFoundError:
        logging.error(f"Credentials file not found at: {credentials_path}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during Google Sheets loading: {e}")
        return False


# --- Load to PostgreSQL ---
def load_to_postgres(
    df: pd.DataFrame, db_config: Dict[str, str], table_name: str = "fashion_products"
) -> bool:
    """
    Loads the DataFrame into a PostgreSQL table. Creates table if not exists, then replaces data.

    Args:
        df (pd.DataFrame): The DataFrame to load.
        db_config (Dict[str, str]): Dictionary with DB connection details
                                     (host, port, dbname, user, password).
        table_name (str): The name of the target table in PostgreSQL.

    Returns:
        bool: True if successful, False otherwise.
    """
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        logging.info(
            f"Successfully connected to PostgreSQL database '{db_config.get('dbname')}'"
        )

        # Define schema based on DataFrame - adjust types as needed
        # Inferring types can be risky, explicit definition is safer in production
        # Example: Use VARCHAR for text, NUMERIC for price, REAL for rating, INTEGER for counts, TIMESTAMPTZ for timestamp
        create_table_query = sql.SQL(
            """
        CREATE TABLE IF NOT EXISTS {table} (
            product_name VARCHAR(255),
            price NUMERIC(10, 2),
            rating REAL,
            num_colors INTEGER,
            size VARCHAR(50),
            gender VARCHAR(50),
            image_url TEXT,
            extracted_at TIMESTAMPTZ
            -- Optionally add a PRIMARY KEY if needed, e.g., on product_name + extracted_at
            -- PRIMARY KEY (product_name, extracted_at)
        );
        """
        ).format(table=sql.Identifier(table_name))
        cursor.execute(create_table_query)
        logging.info(f"Ensured table '{table_name}' exists.")

        # Clear existing data in the table before inserting new data (replace strategy)
        truncate_query = sql.SQL("TRUNCATE TABLE {table};").format(
            table=sql.Identifier(table_name)
        )
        cursor.execute(truncate_query)
        logging.info(f"Truncated table '{table_name}' before loading new data.")

        # Prepare data for insertion (handle NaN/NaT -> None for SQL)
        data_tuples = [
            tuple(x) for x in df.replace({np.nan: None, pd.NaT: None}).to_numpy()
        ]
        if not data_tuples:
            logging.info("No data to insert into PostgreSQL.")
            conn.commit()
            return True

        # Prepare INSERT statement
        cols = sql.SQL(", ").join(map(sql.Identifier, df.columns))
        vals = sql.SQL("(%s" + ", %s" * (len(df.columns) - 1) + ")")

        insert_query = sql.SQL("INSERT INTO {table} ({cols}) VALUES %s").format(
            table=sql.Identifier(table_name), cols=cols
        )

        # Use execute_values for efficient bulk insert
        execute_values(cursor, insert_query.as_string(conn), data_tuples)

        conn.commit()
        logging.info(
            f"Successfully loaded {len(data_tuples)} records into PostgreSQL table '{table_name}'."
        )
        return True

    except psycopg2.Error as e:
        logging.error(f"PostgreSQL error: {e}")
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during PostgreSQL loading: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            logging.info("PostgreSQL connection closed.")


# Example Usage (Optional)
# if __name__ == '__main__':
#     # Create Sample DataFrame
#     sample_data = {
#         'product_name': ['Test Product 1', 'Test Product 2'],
#         'price': [99.99, 149.50],
#         'rating': [4.5, 3.8],
#         'num_colors': [3, 5],
#         'size': ['M', 'L'],
#         'gender': ['Unisex', 'Men'],
#         'image_url': ['url1', 'url2'],
#         'extracted_at': [pd.Timestamp.now(tz='UTC').tz_convert('Asia/Jakarta'), pd.Timestamp.now(tz='UTC').tz_convert('Asia/Jakarta')]
#     }
#     sample_df = pd.DataFrame(sample_data)

# --- Test CSV ---
# csv_filepath = 'test_products.csv'
# load_to_csv(sample_df, csv_filepath)
# Check if test_products.csv is created

# --- Test Google Sheets (Requires setup) ---
# print("\nTesting Google Sheets Load (requires credentials and sheet ID):")
# creds_file = 'google-sheets-api.json' # Replace with your actual path
# sheet_id_val = os.getenv("GOOGLE_SHEET_ID")
# if os.path.exists(creds_file):
#     load_to_gsheets(sample_df, creds_file, sheet_id_val, "TestDataSheet")
# else:
#     print("Skipping Google Sheets test: Credentials or Sheet ID not configured.")


# --- Test PostgreSQL (Requires running DB and config) ---
# print("\nTesting PostgreSQL Load (requires running DB and .env config):")

# db_params = {
#     "host": os.getenv("DB_HOST"),
#     "port": os.getenv("DB_PORT"),
#     "dbname": os.getenv("DB_NAME"),
#     "user": os.getenv("DB_USER"),
#     "password": os.getenv("DB_PASSWORD"),
# }
# # Basic check if config seems present
# if all(db_params.values()):
#      load_to_postgres(sample_df, db_params, table_name="products")
# else:
#      print("Skipping PostgreSQL test: DB configuration not found in .env")
