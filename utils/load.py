import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load to CSV ---
def load_to_csv(df: pd.DataFrame, filename: str):
    """Menyimpan DataFrame ke file CSV."""
    if df.empty:
        logging.warning("DataFrame kosong, tidak ada data untuk disimpan ke CSV.")
        return
    try:
        df.to_csv(filename, index=False, encoding='utf-8')
        logging.info(f"Data berhasil disimpan ke {filename}")
    except IOError as e:
        logging.error(f"Gagal menyimpan data ke CSV {filename}: {e}")
    except Exception as e:
        logging.error(f"Terjadi error tidak terduga saat menyimpan ke CSV: {e}")

# --- Load to Google Sheets ---
def load_to_gsheet(df: pd.DataFrame, sheet_key: str, creds_path: str, worksheet_name: str = "Sheet1"):
    """Menyimpan DataFrame ke Google Sheets."""
    if df.empty:
        logging.warning("DataFrame kosong, tidak ada data untuk disimpan ke Google Sheets.")
        return

    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
        gc = gspread.authorize(creds)

        sh = gc.open_by_key(sheet_key)
        worksheet = None
        try:
            worksheet = sh.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            logging.warning(f"Worksheet '{worksheet_name}' tidak ditemukan. Membuat worksheet baru.")
            worksheet = sh.add_worksheet(title=worksheet_name, rows="1", cols="1") # Ukuran awal kecil

        # Opsi: Hapus data lama sebelum load baru
        # worksheet.clear()
        # logging.info(f"Worksheet '{worksheet_name}' dibersihkan.")

        # Konversi tipe data agar kompatibel dengan JSON serialization Gspread
        df_list = df.astype(str).values.tolist() # Ubah semua jadi string atau handle tipe data spesifik
        header = df.columns.tolist()

        # Update data (header + isi)
        worksheet.update([header] + df_list, 'A1') # Mulai dari cell A1

        logging.info(f"Data berhasil diunggah ke Google Sheet '{sh.title}' -> Worksheet '{worksheet_name}'")

    except FileNotFoundError:
        logging.error(f"File kredensial Google Sheets tidak ditemukan di: {creds_path}")
    except gspread.exceptions.APIError as e:
        logging.error(f"Google Sheets API error: {e}")
    except gspread.exceptions.SpreadsheetNotFound:
        logging.error(f"Google Sheet dengan key '{sheet_key}' tidak ditemukan atau tidak ada akses.")
    except Exception as e:
        logging.error(f"Terjadi error tidak terduga saat load ke Google Sheets: {e}")


# --- Load to PostgreSQL ---
def load_to_postgres(df: pd.DataFrame, db_config: dict, table_name: str):
    """Menyimpan DataFrame ke tabel PostgreSQL."""
    if df.empty:
        logging.warning("DataFrame kosong, tidak ada data untuk disimpan ke PostgreSQL.")
        return

    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        logging.info("Koneksi ke database PostgreSQL berhasil.")

        # --- Pilih Strategi: TRUNCATE + INSERT (contoh sederhana) ---
        # Hati-hati! Ini menghapus semua data di tabel sebelum insert baru.
        truncate_query = sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY;").format(sql.Identifier(table_name))
        cursor.execute(truncate_query)
        logging.info(f"Tabel '{table_name}' berhasil di-truncate.")

        # Buat template INSERT
        cols = df.columns.tolist()
        cols_sql = sql.SQL(', ').join(map(sql.Identifier, cols))
        vals_sql = sql.SQL(', ').join(map(sql.Placeholder, cols)) # Gunakan %s placeholders
        insert_query = sql.SQL("INSERT INTO {} ({}) VALUES %s").format(
            sql.Identifier(table_name),
            cols_sql
        )

        # Konversi DataFrame ke list of tuples
        # Pastikan urutan kolom di df sama dengan `cols`
        data_tuples = [tuple(x) for x in df[cols].to_numpy()]

        # Gunakan execute_values untuk efisiensi
        execute_values(cursor, insert_query.as_string(conn), data_tuples)

        conn.commit()
        logging.info(f"{len(data_tuples)} baris data berhasil dimasukkan ke tabel '{table_name}'.")

    except psycopg2.OperationalError as e:
        logging.error(f"Gagal koneksi ke database PostgreSQL: {e}")
    except psycopg2.Error as e:
        logging.error(f"Database error: {e}")
        if conn:
            conn.rollback() # Batalkan transaksi jika error
    except KeyError as e:
        logging.error(f"Konfigurasi database tidak lengkap: {e}")
    except Exception as e:
        logging.error(f"Terjadi error tidak terduga saat load ke PostgreSQL: {e}")
        if conn:
            conn.rollback()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        logging.info("Koneksi ke database PostgreSQL ditutup.")