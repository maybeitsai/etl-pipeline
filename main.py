# main.py
import os
import logging
from dotenv import load_dotenv
from utils.extract import scrape_all_pages
from utils.transform import transform_data
from utils.load import load_to_csv, load_to_gsheets, load_to_postgres

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables from .env file
load_dotenv()

# Configuration
SOURCE_URL = os.getenv("SOURCE_URL")
MAX_PAGES_TO_SCRAPE = 50
CSV_FILEPATH = os.getenv("CSV_FILEPATH", "products.csv")
GOOGLE_SHEETS_CREDENTIALS_PATH = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "products")

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}
POSTGRES_TABLE_NAME = "products"


def run_pipeline():
    """Runs the full ETL pipeline with pagination."""
    logging.info("Starting ETL pipeline...")

    # --- Extract (dengan Pagination) ---
    logging.info(
        f"Starting scraping from {SOURCE_URL} up to {MAX_PAGES_TO_SCRAPE} pages..."
    )
    # Panggil fungsi scrape_all_pages
    all_extracted_data = scrape_all_pages(SOURCE_URL, MAX_PAGES_TO_SCRAPE)

    if not all_extracted_data:
        logging.warning("No data extracted from any page. Pipeline finished.")
        return

    logging.info(
        f"Extraction complete. Total items extracted: {len(all_extracted_data)}"
    )

    # --- Transform ---
    logging.info("Transforming extracted data...")
    # Gunakan data gabungan dari semua halaman
    transformed_df = transform_data(all_extracted_data)
    if transformed_df is None:
        logging.error("Transformation failed. Aborting pipeline.")
        return
    if transformed_df.empty:
        logging.warning(
            "Transformation resulted in an empty DataFrame. Check extraction/transformation logic."
        )
        return

    logging.info(f"Transformation complete. Processed {len(transformed_df)} products.")
    logging.info(
        "Sample transformed data:\n" + transformed_df.head().to_string()
    )  # Tampilkan head()

    # --- Load ---
    logging.info("Starting data loading phase...")

    # Load to CSV
    if load_to_csv(transformed_df, CSV_FILEPATH):
        logging.info(f"Successfully loaded data to {CSV_FILEPATH}")
    else:
        logging.error(f"Failed to load data to {CSV_FILEPATH}")

    # Load to Google Sheets
    if GOOGLE_SHEET_ID and os.path.exists(GOOGLE_SHEETS_CREDENTIALS_PATH):
        if load_to_gsheets(
            transformed_df,
            GOOGLE_SHEETS_CREDENTIALS_PATH,
            GOOGLE_SHEET_ID,
            WORKSHEET_NAME,
        ):
            logging.info(
                f"Successfully loaded data to Google Sheet ID: {GOOGLE_SHEET_ID}"
            )
        else:
            logging.error("Failed to load data to Google Sheets.")
    else:
        logging.warning(
            "Skipping Google Sheets load: Sheet ID not set in .env or credentials file missing."
        )

    # Load to PostgreSQL
    if all(val is not None for val in DB_CONFIG.values()):
        if load_to_postgres(transformed_df, DB_CONFIG, POSTGRES_TABLE_NAME):
            logging.info(
                f"Successfully loaded data to PostgreSQL table: {POSTGRES_TABLE_NAME}"
            )
        else:
            logging.error("Failed to load data to PostgreSQL.")
    else:
        logging.warning(
            "Skipping PostgreSQL load: Database configuration is incomplete in .env."
        )

    logging.info("ETL pipeline finished.")


if __name__ == "__main__":
    run_pipeline()
