# tests/test_load.py

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock, mock_open, ANY
import os
import logging
import gspread # Import untuk exception
import psycopg2 # Import untuk exception

# Pastikan path ke utils benar
from utils.load import load_to_csv, load_to_gsheet, load_to_postgres

@pytest.fixture
def sample_dataframe():
    """Fixture untuk menyediakan DataFrame sampel."""
    return pd.DataFrame({
        'name': ['Test Product 1', 'Test Product 2'],
        'category': ['Test Category', 'Test Category'],
        'price': [10000.0, 25000.0],
        'url': ['/test1', '/test2'],
        'scraped_at': [datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 12, 5, 0)]
    })

# --- Test Load to CSV ---

def test_load_to_csv_success(sample_dataframe, tmp_path):
    """Test load ke CSV berhasil menggunakan temporary path."""
    # tmp_path adalah fixture pytest yang menyediakan direktori sementara unik
    output_file = tmp_path / "test_output.csv"
    load_to_csv(sample_dataframe, str(output_file))

    # Periksa apakah file dibuat
    assert output_file.is_file()

    # Periksa isinya (opsional tapi bagus)
    df_read = pd.read_csv(output_file)
    # Perlu penyesuaian tipe data saat membaca (misal, datetime jadi string)
    # Cara mudah: bandingkan setelah konversi ke list of dict
    expected_dict = sample_dataframe.astype(str).to_dict('records')
    read_dict = df_read.astype(str).to_dict('records')
    assert read_dict == expected_dict

def test_load_to_csv_empty_df(tmp_path, caplog):
    """Test load ke CSV dengan DataFrame kosong."""
    output_file = tmp_path / "empty_output.csv"
    empty_df = pd.DataFrame()

    with caplog.at_level(logging.WARNING):
        load_to_csv(empty_df, str(output_file))

    assert not output_file.exists() # File tidak boleh dibuat
    assert "DataFrame kosong, tidak ada data untuk disimpan ke CSV." in caplog.text

@patch("pandas.DataFrame.to_csv")
def test_load_to_csv_io_error(mock_to_csv, sample_dataframe, caplog):
    """Test penanganan IOError saat menyimpan CSV."""
    mock_to_csv.side_effect = IOError("Permission denied")

    with caplog.at_level(logging.ERROR):
        load_to_csv(sample_dataframe, "forbidden_path/output.csv")

    assert "Gagal menyimpan data ke CSV" in caplog.text
    assert "Permission denied" in caplog.text
    mock_to_csv.assert_called_once()

# --- Test Load to Google Sheets ---

# Patch object yang akan dibuat di dalam fungsi load_to_gsheet
@patch('utils.load.gspread.authorize')
@patch('utils.load.Credentials.from_service_account_file')
def test_load_to_gsheet_success(mock_creds_from_file, mock_gspread_authorize, sample_dataframe):
    """Test load ke Google Sheets berhasil (dengan mock)."""
    # Setup mock objects
    mock_gc = MagicMock() # Mock gspread client
    mock_sh = MagicMock() # Mock spreadsheet
    mock_ws = MagicMock() # Mock worksheet

    mock_gspread_authorize.return_value = mock_gc
    mock_gc.open_by_key.return_value = mock_sh
    mock_sh.worksheet.return_value = mock_ws
    mock_sh.title = "Mock Sheet" # Beri nama untuk logging

    creds_path = "dummy/creds.json"
    sheet_key = "dummy_sheet_key"
    worksheet_name = "TestSheet"

    load_to_gsheet(sample_dataframe, sheet_key, creds_path, worksheet_name)

    # Verifikasi pemanggilan
    mock_creds_from_file.assert_called_once_with(creds_path, scopes=ANY)
    mock_gspread_authorize.assert_called_once()
    mock_gc.open_by_key.assert_called_once_with(sheet_key)
    mock_sh.worksheet.assert_called_once_with(worksheet_name)

    # Cek apakah worksheet.update dipanggil dengan data yang benar
    # Header + data rows, semua dikonversi ke string
    expected_header = sample_dataframe.columns.tolist()
    expected_data = sample_dataframe.astype(str).values.tolist()
    expected_call_arg = [expected_header] + expected_data
    mock_ws.update.assert_called_once_with(expected_call_arg, 'A1')

@patch('utils.load.gspread.authorize')
@patch('utils.load.Credentials.from_service_account_file')
def test_load_to_gsheet_worksheet_not_found(mock_creds, mock_auth, sample_dataframe, caplog):
    """Test ketika worksheet tidak ditemukan dan dibuat baru."""
    mock_gc = MagicMock()
    mock_sh = MagicMock()
    mock_ws_new = MagicMock() # Worksheet baru yg dibuat

    mock_auth.return_value = mock_gc
    mock_gc.open_by_key.return_value = mock_sh
    # Simulasikan worksheet tidak ditemukan, lalu add_worksheet mengembalikan mock baru
    mock_sh.worksheet.side_effect = gspread.exceptions.WorksheetNotFound
    mock_sh.add_worksheet.return_value = mock_ws_new
    mock_sh.title = "Mock Sheet"

    with caplog.at_level(logging.WARNING):
        load_to_gsheet(sample_dataframe, "key", "creds", "NewSheet")

    assert "Worksheet 'NewSheet' tidak ditemukan. Membuat worksheet baru." in caplog.text
    mock_sh.add_worksheet.assert_called_once_with(title="NewSheet", rows=ANY, cols=ANY)
    # Pastikan update dipanggil pada worksheet baru
    mock_ws_new.update.assert_called_once()

@patch('utils.load.Credentials.from_service_account_file')
def test_load_to_gsheet_creds_not_found(mock_creds, sample_dataframe, caplog):
    """Test penanganan jika file kredensial tidak ditemukan."""
    mock_creds.side_effect = FileNotFoundError("Creds file missing")

    with caplog.at_level(logging.ERROR):
        load_to_gsheet(sample_dataframe, "key", "invalid/path.json", "Sheet1")

    assert "File kredensial Google Sheets tidak ditemukan" in caplog.text
    assert "invalid/path.json" in caplog.text

# --- Test Load to PostgreSQL ---

# Patch koneksi dan cursor
@patch('utils.load.psycopg2.connect')
def test_load_to_postgres_success(mock_connect, sample_dataframe):
    """Test load ke PostgreSQL berhasil (dengan mock)."""
    # Setup mock connection dan cursor
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor

    db_config = {'host': 'h', 'port': 'p', 'dbname': 'd', 'user': 'u', 'password': 'pw'}
    table_name = "test_products"

    load_to_postgres(sample_dataframe, db_config, table_name)

    # Verifikasi koneksi dibuat dengan config yang benar
    mock_connect.assert_called_once_with(**db_config)
    mock_conn.cursor.assert_called_once()

    # Verifikasi TRUNCATE dipanggil (sesuai implementasi load_to_postgres kita)
    # ANY karena objek sql.SQL sulit di-assert secara langsung
    mock_cursor.execute.assert_any_call(ANY) # Cek panggilan TRUNCATE

    # Verifikasi execute_values dipanggil (sesuai implementasi kita)
    # ANY karena objek sql.SQL sulit di-assert
    # Cek bahwa argumen kedua adalah list of tuples
    from psycopg2.extras import execute_values
    # Cari panggilan ke execute_values (mungkin perlu mock execute_values secara spesifik)
    # Atau cek panggilan cursor.execute yang kedua (jika tidak pakai execute_values)

    # Verifikasi commit dan close dipanggil
    mock_conn.commit.assert_called_once()
    mock_cursor.close.assert_called_once()
    mock_conn.close.assert_called_once()


@patch('utils.load.psycopg2.connect')
def test_load_to_postgres_connection_error(mock_connect, sample_dataframe, caplog):
    """Test penanganan error koneksi PostgreSQL."""
    mock_connect.side_effect = psycopg2.OperationalError("Connection failed")
    db_config = {'host': 'h', 'port': 'p', 'dbname': 'd', 'user': 'u', 'password': 'pw'}

    with caplog.at_level(logging.ERROR):
        load_to_postgres(sample_dataframe, db_config, "products")

    assert "Gagal koneksi ke database PostgreSQL" in caplog.text
    assert "Connection failed" in caplog.text
    # Pastikan commit/close tidak dipanggil jika koneksi gagal
    mock_connect.return_value.commit.assert_not_called()
    mock_connect.return_value.close.assert_not_called() # Mungkin dipanggil di finally, ok

@patch('utils.load.psycopg2.connect')
def test_load_to_postgres_execute_error(mock_connect, sample_dataframe, caplog):
    """Test penanganan error saat eksekusi query (misal, insert)."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor
    # Simulasikan error saat execute_values atau execute
    mock_cursor.execute.side_effect = [
        None, # Sukses untuk TRUNCATE
        psycopg2.Error("Insert failed") # Error untuk INSERT
    ]
    # Jika pakai execute_values, mock itu:
    # with patch('utils.load.execute_values') as mock_exec_values:
    #     mock_exec_values.side_effect = psycopg2.Error("Insert failed")

    db_config = {'host': 'h', 'port': 'p', 'dbname': 'd', 'user': 'u', 'password': 'pw'}

    with caplog.at_level(logging.ERROR):
        load_to_postgres(sample_dataframe, db_config, "products")

    assert "Database error" in caplog.text
    assert "Insert failed" in caplog.text
    # Pastikan rollback dipanggil
    mock_conn.rollback.assert_called_once()
    # Commit seharusnya tidak dipanggil
    mock_conn.commit.assert_not_called()
    mock_cursor.close.assert_called_once()
    mock_conn.close.assert_called_once()