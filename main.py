import os
from dotenv import load_dotenv
import logging

# Import fungsi dari modul utils
from utils.extract import extract_products
from utils.transform import transform_data
from utils.load import load_to_csv, load_to_gsheet, load_to_postgres

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Muat variabel lingkungan dari .env
    load_dotenv()
    logging.info("Variabel lingkungan dimuat.")

    # Konfigurasi dari environment variables
    target_url = "https://fashion-studio.dicoding.dev" # Atau ambil dari env jika perlu
    csv_output_file = "products.csv"
    gsheet_key = os.getenv('GSPREAD_SHEET_KEY')
    gsheet_creds_path = os.getenv('GSPREAD_SERVICE_ACCOUNT_FILE')
    gsheet_worksheet_name = "Produk Kompetitor" # Nama worksheet yang diinginkan

    db_config = {
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT'),
        'dbname': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
    }
    db_table_name = "products" # Sesuaikan dengan nama tabel Anda

    logging.info("--- Memulai Pipeline ETL ---")

    # 1. Extract
    logging.info(f"Memulai ekstraksi dari {target_url}")
    raw_data = extract_products(target_url)
    if not raw_data:
        logging.error("Ekstraksi gagal atau tidak menghasilkan data. Pipeline berhenti.")
        return
    logging.info(f"Ekstraksi berhasil, {len(raw_data)} produk mentah ditemukan.")

    # 2. Transform
    logging.info("Memulai transformasi data.")
    transformed_df = transform_data(raw_data)
    if transformed_df.empty:
        logging.error("Transformasi gagal atau menghasilkan DataFrame kosong. Pipeline berhenti.")
        return
    logging.info("Transformasi data berhasil.")
    # logging.info("Preview data setelah transformasi:\n" + transformed_df.head().to_string())


    # 3. Load
    logging.info("Memulai proses load data.")

    # Load ke CSV
    load_to_csv(transformed_df, csv_output_file)

    # Load ke Google Sheets
    if gsheet_key and gsheet_creds_path and os.path.exists(gsheet_creds_path):
        load_to_gsheet(transformed_df, gsheet_key, gsheet_creds_path, gsheet_worksheet_name)
    else:
        logging.warning("Skipping load ke Google Sheets: GSPREAD_SHEET_KEY atau GSPREAD_SERVICE_ACCOUNT_FILE tidak valid/tidak ditemukan.")

    # Load ke PostgreSQL
    # Cek apakah semua konfigurasi DB ada
    if all(db_config.values()):
        load_to_postgres(transformed_df, db_config, db_table_name)
    else:
        logging.warning("Skipping load ke PostgreSQL: Konfigurasi database tidak lengkap di .env.")

    logging.info("--- Pipeline ETL Selesai ---")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Terjadi error tidak terduga di pipeline utama: {e}", exc_info=True)