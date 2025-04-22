# tests/test_load.py
"""
Unit tests for the utils.load module.
"""

import logging
import os
from unittest.mock import MagicMock, call, patch

import gspread
import numpy as np # Import numpy for NaN
import pandas as pd
import psycopg2
import pytest
from gspread.exceptions import GSpreadException, WorksheetNotFound
from psycopg2 import Error as Psycopg2Error
from psycopg2 import sql
from psycopg2.extras import execute_values # Ensure execute_values is imported for patching

# Assuming utils is importable from the project root
from utils import load
from utils.constants import WORKSHEET_NAME

# --- Sample Data Fixtures ---
@pytest.fixture
def sample_df():
    """Provides a sample DataFrame for load tests."""
    data = {
        "title": ["Product A", "Product B"],
        "price": [150000.0, 2500000.50],
        "rating": [4.5, 4.8],
        "colors": [3, 1],
        "size": ["M", "XL"],
        "gender": ["Unisex", "Men"],
        "image_url": ["url_a", "url_b"],
        "timestamp": pd.to_datetime(
            ["2024-01-15 10:00:00+07:00", "2024-01-15 11:00:00+07:00"],
            utc=True
        ).tz_convert("Asia/Jakarta"),
    }
    # Ensure correct dtypes including Int64 for nullable integers if needed
    df = pd.DataFrame(data).astype({
        "title": str,
        "price": float,
        "rating": float,
        "colors": 'Int64', # Use nullable integer type
        "size": str,
        "gender": str,
        "image_url": str,
        # Timestamp already correct dtype from creation
    })
    return df

@pytest.fixture
def empty_df():
    """Provides an empty DataFrame."""
    return pd.DataFrame()

@pytest.fixture
def mock_db_config():
    """Provides mock database configuration."""
    return {"host": "localhost", "port": "5432", "dbname": "testdb", "user": "testuser", "password": "password"}


# --- Tests for load_to_csv ---

@patch("utils.load.os.makedirs")
@patch("pandas.DataFrame.to_csv")
def test_load_to_csv_success(mock_to_csv, mock_makedirs, sample_df, tmp_path):
    """Tests successful saving to CSV."""
    # Arrange
    filepath = tmp_path / "output" / "data.csv"

    # Act
    result = load.load_to_csv(sample_df, str(filepath))

    # Assert
    assert result is True
    mock_makedirs.assert_called_once_with(tmp_path / "output", exist_ok=True)
    mock_to_csv.assert_called_once_with(str(filepath), index=False, encoding="utf-8")

@patch("utils.load.os.makedirs")
@patch("pandas.DataFrame.to_csv")
def test_load_to_csv_no_subdir(mock_to_csv, mock_makedirs, sample_df, tmp_path):
    """Tests successful saving to CSV in the root temp dir."""
    # Arrange
    filepath = tmp_path / "data.csv"

    # Act
    result = load.load_to_csv(sample_df, str(filepath))

    # Assert
    assert result is True
    mock_makedirs.assert_not_called() # No subdir creation needed
    mock_to_csv.assert_called_once_with(str(filepath), index=False, encoding="utf-8")

@patch("utils.load.os.makedirs") # Need to patch makedirs even if not expected
@patch("pandas.DataFrame.to_csv")
def test_load_to_csv_empty_df(mock_to_csv, mock_makedirs, empty_df, tmp_path, caplog):
    """Tests skipping save for an empty DataFrame."""
    # Arrange
    filepath = tmp_path / "empty.csv"
    caplog.set_level(logging.WARNING)

    # Act
    result = load.load_to_csv(empty_df, str(filepath))

    # Assert
    assert result is True
    mock_makedirs.assert_not_called()
    mock_to_csv.assert_not_called()
    assert f"DataFrame is empty. Skipping CSV save to {filepath}" in caplog.text

@patch("utils.load.os.makedirs") # Need to patch makedirs
@patch("pandas.DataFrame.to_csv", side_effect=IOError("Disk full"))
def test_load_to_csv_io_error(mock_to_csv, mock_makedirs, sample_df, tmp_path, caplog):
    """Tests handling of IOError during CSV save."""
    # Arrange
    filepath = tmp_path / "error.csv"
    caplog.set_level(logging.ERROR)

    # Act
    result = load.load_to_csv(sample_df, str(filepath))

    # Assert
    assert result is False
    mock_makedirs.assert_called_once() # Directory creation is attempted
    mock_to_csv.assert_called_once()
    assert f"Error saving data to CSV file {filepath}: Disk full" in caplog.text

@patch("utils.load.os.makedirs") # Need to patch makedirs
@patch("pandas.DataFrame.to_csv", side_effect=OSError("Permission denied"))
def test_load_to_csv_os_error(mock_to_csv, mock_makedirs, sample_df, tmp_path, caplog):
    """Tests handling of OSError during CSV save."""
    # Arrange
    filepath = tmp_path / "os_error.csv"
    caplog.set_level(logging.ERROR)

    # Act
    result = load.load_to_csv(sample_df, str(filepath))

    # Assert
    assert result is False
    mock_makedirs.assert_called_once()
    mock_to_csv.assert_called_once()
    assert f"Error saving data to CSV file {filepath}: Permission denied" in caplog.text

@patch("utils.load.os.makedirs") # Need to patch makedirs
@patch("pandas.DataFrame.to_csv", side_effect=Exception("Unexpected pandas error"))
def test_load_to_csv_unexpected_error(mock_to_csv, mock_makedirs, sample_df, tmp_path, caplog):
    """Tests handling of unexpected exceptions during CSV save."""
    # Arrange
    filepath = tmp_path / "unexpected.csv"
    caplog.set_level(logging.ERROR)

    # Act
    result = load.load_to_csv(sample_df, str(filepath))

    # Assert
    assert result is False
    mock_makedirs.assert_called_once()
    mock_to_csv.assert_called_once()
    assert f"An unexpected error occurred during CSV saving to {filepath}" in caplog.text
    assert "Unexpected pandas error" in caplog.text

# --- Tests for _prepare_gsheets_data ---
def test_prepare_gsheets_data_success(sample_df):
    """Tests successful preparation of data for gspread."""
    # Arrange
    df = sample_df.copy()
    # Add NaN/NaT for testing replacement
    df.loc[0, 'rating'] = np.nan
    df.loc[1, 'timestamp'] = pd.NaT
    df.loc[0, 'colors'] = pd.NA # Use pandas NA for nullable Int

    # Act
    result_list = load._prepare_gsheets_data(df)

    # Assert
    # Check header row
    assert result_list[0] == list(df.columns)
    # Check first data row (index 0)
    assert result_list[1][0] == "Product A"  # title
    assert result_list[1][2] is None        # rating was NaN
    assert result_list[1][3] is None        # colors was pd.NA
    assert result_list[1][-1] == "2024-01-15 10:00:00+0700" # timestamp original
    # Check second data row (index 1)
    assert result_list[2][0] == "Product B" # title
    assert result_list[2][2] == 4.8         # rating original
    assert result_list[2][3] == 1           # colors original
    assert result_list[2][-1] is None       # timestamp was NaT

def test_prepare_gsheets_data_no_timestamp(sample_df):
    """Tests preparation when timestamp column is missing."""
    # Arrange
    df_no_ts = sample_df.drop(columns=['timestamp'])

    # Act
    result_list = load._prepare_gsheets_data(df_no_ts)

    # Assert
    assert list(df_no_ts.columns) == result_list[0]
    assert len(result_list) == len(df_no_ts) + 1
    # Check no timestamp formatting error occurred

def test_prepare_gsheets_data_timestamp_not_datetime(sample_df):
    """Tests preparation when timestamp column is not datetime."""
    # Arrange
    df_wrong_ts = sample_df.copy()
    df_wrong_ts['timestamp'] = ["today", "yesterday"] # String type

    # Act
    result_list = load._prepare_gsheets_data(df_wrong_ts)

    # Assert
    assert list(df_wrong_ts.columns) == result_list[0]
    assert result_list[1][-1] == "today" # Timestamp is treated as object, not formatted
    assert result_list[2][-1] == "yesterday"

# --- Tests for load_to_gsheets ---

@patch("utils.load._prepare_gsheets_data") # Patch helper
@patch("utils.load.gspread.authorize")
@patch("utils.load.Credentials.from_service_account_file")
def test_load_to_gsheets_success_worksheet_exists(mock_creds, mock_authorize, mock_prepare, sample_df):
    """Tests successful load to an existing Google Sheet worksheet."""
    # Arrange
    mock_gc = MagicMock()
    mock_spreadsheet = MagicMock()
    mock_worksheet = MagicMock()
    mock_authorize.return_value = mock_gc
    mock_gc.open_by_key.return_value = mock_spreadsheet
    mock_spreadsheet.worksheet.return_value = mock_worksheet

    prepared_data = [["col1", "col2"], ["a", 1], ["b", 2]] # Mock prepared data
    mock_prepare.return_value = prepared_data

    credentials_path = "fake_creds.json"
    sheet_id = "fake_sheet_id"
    worksheet_name = "TestData"

    # Act
    result = load.load_to_gsheets(sample_df, credentials_path, sheet_id, worksheet_name)

    # Assert
    assert result is True
    mock_creds.assert_called_once_with(credentials_path, scopes=["https://www.googleapis.com/auth/spreadsheets"])
    mock_authorize.assert_called_once_with(mock_creds.return_value)
    mock_gc.open_by_key.assert_called_once_with(sheet_id)
    mock_prepare.assert_called_once_with(sample_df) # Verify helper called
    mock_spreadsheet.worksheet.assert_called_once_with(worksheet_name)
    mock_spreadsheet.add_worksheet.assert_not_called()
    mock_worksheet.clear.assert_called_once()
    mock_worksheet.update.assert_called_once_with(prepared_data, value_input_option="USER_ENTERED")


@patch("utils.load._prepare_gsheets_data") # Patch helper
@patch("utils.load.gspread.authorize")
@patch("utils.load.Credentials.from_service_account_file")
def test_load_to_gsheets_success_create_worksheet(mock_creds, mock_authorize, mock_prepare, sample_df):
    """Tests successful load when the worksheet needs to be created."""
    # Arrange
    mock_gc = MagicMock()
    mock_spreadsheet = MagicMock()
    mock_new_worksheet = MagicMock()
    mock_authorize.return_value = mock_gc
    mock_gc.open_by_key.return_value = mock_spreadsheet
    mock_spreadsheet.worksheet.side_effect = WorksheetNotFound
    mock_spreadsheet.add_worksheet.return_value = mock_new_worksheet

    prepared_data = [["col1", "col2"], ["a", 1]] # Mock prepared data
    mock_prepare.return_value = prepared_data

    credentials_path = "fake_creds.json"
    sheet_id = "fake_sheet_id"
    worksheet_name = "NewSheet"

    # Act
    result = load.load_to_gsheets(sample_df, credentials_path, sheet_id, worksheet_name)

    # Assert
    assert result is True
    mock_prepare.assert_called_once_with(sample_df)
    mock_spreadsheet.worksheet.assert_called_once_with(worksheet_name)
    mock_spreadsheet.add_worksheet.assert_called_once_with(title=worksheet_name, rows=1, cols=1)
    mock_new_worksheet.clear.assert_called_once()
    mock_new_worksheet.update.assert_called_once_with(prepared_data, value_input_option="USER_ENTERED")


@patch("utils.load.gspread.authorize")
@patch("utils.load.Credentials.from_service_account_file")
def test_load_to_gsheets_empty_df(mock_creds, mock_authorize, empty_df, caplog):
    """Tests skipping Google Sheets load for an empty DataFrame."""
    # Arrange
    credentials_path = "fake_creds.json"
    sheet_id = "fake_sheet_id"
    worksheet_name = "EmptySheet"
    caplog.set_level(logging.WARNING)

    # Act
    result = load.load_to_gsheets(empty_df, credentials_path, sheet_id, worksheet_name)

    # Assert
    assert result is True
    mock_creds.assert_not_called()
    mock_authorize.assert_not_called()
    assert f"DataFrame is empty. Skipping Google Sheets load to sheet ID {sheet_id}, worksheet '{worksheet_name}'." in caplog.text


@patch("utils.load.gspread.authorize") # Patch authorize to avoid API calls
@patch("utils.load.Credentials.from_service_account_file", side_effect=FileNotFoundError("Creds not found"))
def test_load_to_gsheets_credentials_not_found(mock_creds, mock_authorize, sample_df, caplog):
    """Tests handling FileNotFoundError for credentials."""
    # Arrange
    credentials_path = "nonexistent_creds.json"
    sheet_id = "fake_sheet_id"
    worksheet_name = "TestData"
    caplog.set_level(logging.ERROR)

    # Act
    result = load.load_to_gsheets(sample_df, credentials_path, sheet_id, worksheet_name)

    # Assert
    assert result is False
    mock_creds.assert_called_once_with(credentials_path, scopes=["https://www.googleapis.com/auth/spreadsheets"])
    mock_authorize.assert_not_called() # Authorize is not reached
    assert f"Credentials file not found at: {credentials_path}" in caplog.text


@patch("utils.load.gspread.authorize", side_effect=GSpreadException("API Error"))
@patch("utils.load.Credentials.from_service_account_file")
def test_load_to_gsheets_gspread_error(mock_creds, mock_authorize, sample_df, caplog):
    """Tests handling of GSpreadException."""
    # Arrange
    credentials_path = "fake_creds.json"
    sheet_id = "fake_sheet_id"
    worksheet_name = "TestData"
    caplog.set_level(logging.ERROR)

    # Act
    result = load.load_to_gsheets(sample_df, credentials_path, sheet_id, worksheet_name)

    # Assert
    assert result is False
    mock_creds.assert_called_once()
    mock_authorize.assert_called_once()
    assert "Google Sheets API or client error: API Error" in caplog.text


@patch("utils.load.gspread.authorize")
@patch("utils.load.Credentials.from_service_account_file")
@patch("utils.load._prepare_gsheets_data", side_effect=ValueError("Bad data format"))
def test_load_to_gsheets_value_error(mock_prepare_data, mock_creds, mock_authorize, sample_df, caplog):
    """Tests handling of ValueError during GSheets data preparation."""
    # Arrange
    mock_gc = MagicMock() # Mock gc needed as it's used before prepare_data is called
    mock_authorize.return_value = mock_gc
    credentials_path = "fake_creds.json"
    sheet_id = "fake_sheet_id"
    worksheet_name = "TestData"
    caplog.set_level(logging.ERROR)

    # Act
    result = load.load_to_gsheets(sample_df, credentials_path, sheet_id, worksheet_name)

    # Assert
    assert result is False
    mock_creds.assert_called_once()
    mock_authorize.assert_called_once()
    mock_prepare_data.assert_called_once_with(sample_df)
    assert "Value error during Google Sheets operation: Bad data format" in caplog.text


@patch("utils.load.gspread.authorize")
@patch("utils.load.Credentials.from_service_account_file")
def test_load_to_gsheets_unexpected_error(mock_creds, mock_authorize, sample_df, caplog):
    """Tests handling of unexpected exceptions during GSheets operations."""
    # Arrange
    mock_gc = MagicMock()
    mock_authorize.return_value = mock_gc
    mock_gc.open_by_key.side_effect = Exception("Something else went wrong")
    credentials_path = "fake_creds.json"
    sheet_id = "fake_sheet_id"
    worksheet_name = "TestData"
    caplog.set_level(logging.ERROR)

    # Act
    result = load.load_to_gsheets(sample_df, credentials_path, sheet_id, worksheet_name)

    # Assert
    assert result is False
    mock_creds.assert_called_once()
    mock_authorize.assert_called_once()
    mock_gc.open_by_key.assert_called_once_with(sheet_id)
    assert "An unexpected error occurred during Google Sheets loading" in caplog.text
    assert "Something else went wrong" in caplog.text


# --- Tests for _get_postgres_schema ---

def test_get_postgres_schema_standard_types():
    """Tests schema generation for standard pandas dtypes."""
    # Arrange
    df = pd.DataFrame({
        'col_str': ['a', 'b'],
        'col_int': [1, 2],
        'col_float': [1.1, 2.2],
        'col_bool': [True, False],
        'col_dt_naive': pd.to_datetime(['2024-01-01', '2024-01-02']),
        'price': [10.5, 20.0] # Special handling for price
    }).astype({
        'col_str': 'object',
        'col_int': 'int64',
        'col_float': 'float64',
        'col_bool': 'bool',
        'price': 'float64'
    })
    # Set dtype explicitly for datetime
    df['col_dt_naive'] = df['col_dt_naive'].astype('datetime64[ns]')

    # Act
    schema_sql = load._get_postgres_schema(df)
    # Need a dummy cursor to render the SQL for comparison
    dummy_conn = MagicMock()
    dummy_cursor = MagicMock()
    dummy_cursor.connection = dummy_conn
    rendered_sql = schema_sql.as_string(dummy_cursor)

    # Assert
    expected = '"col_str" TEXT, "col_int" INTEGER, "col_float" REAL, "col_bool" BOOLEAN, "col_dt_naive" TIMESTAMP, "price" NUMERIC(12, 2)'
    assert rendered_sql == expected

def test_get_postgres_schema_datetime_aware():
    """Tests schema generation for timezone-aware datetime."""
    # Arrange
    df = pd.DataFrame({
        'col_dt_aware': pd.to_datetime(['2024-01-01 10:00:00+07:00'], utc=True).tz_convert('Asia/Jakarta')
    })

    # Act
    schema_sql = load._get_postgres_schema(df)
    dummy_conn = MagicMock()
    dummy_cursor = MagicMock()
    dummy_cursor.connection = dummy_conn
    rendered_sql = schema_sql.as_string(dummy_cursor)

    # Assert
    assert rendered_sql == '"col_dt_aware" TIMESTAMPTZ'

def test_get_postgres_schema_unmapped_type(caplog):
    """Tests schema generation fallback for unmapped dtypes."""
    # Arrange
    df = pd.DataFrame({'col_complex': [1+2j, 3+4j]}) # Complex dtype
    caplog.set_level(logging.WARNING)

    # Act
    schema_sql = load._get_postgres_schema(df)
    dummy_conn = MagicMock()
    dummy_cursor = MagicMock()
    dummy_cursor.connection = dummy_conn
    rendered_sql = schema_sql.as_string(dummy_cursor)

    # Assert
    assert rendered_sql == '"col_complex" TEXT'
    assert "Unmapped dtype 'complex128' for column 'col_complex'. Defaulting to TEXT." in caplog.text


# --- Tests for load_to_postgres ---

# TEST YANG DISEMPURNAKAN UNTUK MENCAKUP BARIS 262-268
@patch("utils.load.psycopg2.connect")
@patch("utils.load._get_postgres_schema")
@patch("utils.load.execute_values")
def test_load_to_postgres_success_covers_final_commit_log_return(mock_execute_values, mock_get_schema, mock_connect, sample_df, mock_db_config, caplog):
    """
    Test pemuatan sukses, secara spesifik memastikan cakupan untuk commit akhir,
    pesan log, dan pernyataan return (baris 262-268).
    """
    # Arrange
    mock_conn = MagicMock(name="MockConnection")
    mock_cursor = MagicMock(name="MockCursor")
    mock_connect.return_value = mock_conn
    # Konfigurasi context manager untuk cursor
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    # Pastikan __exit__ tidak menimbulkan error (penting untuk 'with' statement)
    mock_conn.cursor.return_value.__exit__.return_value = None

    # Mock generasi skema
    mock_schema_sql = load._get_postgres_schema(sample_df) # Gunakan skema asli untuk akurasi
    mock_get_schema.return_value = mock_schema_sql

    table_name = "products_final_check"
    # Tangkap log level INFO, penting untuk baris target
    caplog.set_level(logging.INFO)

    # Siapkan data tuple yang diharapkan *setelah* diproses
    # Ini membantu memverifikasi jumlah dalam pesan log secara akurat
    df_prepared_expected = sample_df.astype(object).where(pd.notnull(sample_df), None)
    expected_data_tuples = [tuple(x) for x in df_prepared_expected.to_numpy()]
    expected_row_count = len(expected_data_tuples)
    assert expected_row_count > 0, "Sample DataFrame should produce data tuples"

    # Act
    result = load.load_to_postgres(sample_df, mock_db_config, table_name)

    # Assert
    # 1. Periksa nilai return (mencakup baris 268)
    assert result is True, "Fungsi harus mengembalikan True pada sukses"

    # 2. Verifikasi mock dipanggil seperti yang diharapkan hingga titik commit
    mock_connect.assert_called_once_with(**mock_db_config)
    mock_get_schema.assert_called_once_with(sample_df)
    # Pastikan CREATE TABLE dan TRUNCATE dipanggil
    mock_cursor.execute.assert_any_call(sql.SQL(
        "CREATE TABLE IF NOT EXISTS {table} ({schema});"
    ).format(table=sql.Identifier(table_name), schema=mock_schema_sql))
    mock_cursor.execute.assert_any_call(sql.SQL(
        "TRUNCATE TABLE {table};"
    ).format(table=sql.Identifier(table_name)))
    # Verifikasi execute_values dipanggil (penting sebelum commit akhir)
    mock_execute_values.assert_called_once()

    # 3. Verifikasi commit dipanggil (mencakup baris 262)
    try:
        mock_conn.commit.assert_called_once()
    except AssertionError as e:
        print(f"DEBUG: Jumlah panggilan mock commit: {mock_conn.commit.call_count}")
        print(f"DEBUG: Mock commit calls: {mock_conn.commit.call_args_list}")
        raise e

    # 4. Verifikasi pesan log sukses spesifik (mencakup baris 263-267)
    final_log_message = f"Successfully loaded {expected_row_count} records into PostgreSQL table '{table_name}'. Committed."
    assert final_log_message in caplog.text, f"Log sukses akhir '{final_log_message}' tidak ditemukan di caplog.text. Logs: {caplog.text}"
    log_found_in_records = any(
        rec.levelname == "INFO" and rec.message == final_log_message
        for rec in caplog.records
    )
    assert log_found_in_records, f"Log sukses akhir '{final_log_message}' tidak ditemukan di caplog.records"

    # 5. Verifikasi rollback tidak dipanggil
    mock_conn.rollback.assert_not_called()

    # 6. Verifikasi koneksi ditutup
    mock_conn.close.assert_called_once()


# TEST BARU UNTUK JALUR 'if not data_tuples:' (Baris 244-246)
@patch("utils.load.psycopg2.connect")
@patch("utils.load._get_postgres_schema")
# --- Patch Kunci: Paksa df_prepared.to_numpy() mengembalikan list kosong ---
@patch("pandas.DataFrame.to_numpy")
# --- Patch juga execute_values untuk memastikan TIDAK dipanggil ---
@patch("utils.load.execute_values")
def test_load_to_postgres_force_empty_tuples_path(mock_execute_values, mock_to_numpy, mock_get_schema, mock_connect, sample_df, mock_db_config, caplog):
    """
    Test jalur spesifik (kemungkinan tidak terjangkau) di mana data_tuples menjadi kosong
    setelah memproses DataFrame yang tidak kosong dengan memaksa df_prepared.to_numpy()
    mengembalikan list kosong. Mencakup baris 244-246.
    """
    # Arrange
    # Kita perlu df yang tidak kosong untuk melewati pemeriksaan awal
    assert not sample_df.empty

    mock_conn = MagicMock(name="mock_connection")
    mock_cursor = MagicMock(name="mock_cursor")
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_conn.cursor.return_value.__exit__.return_value = None # Handle exit context manager
    mock_schema_sql = sql.SQL('"dummy" TEXT') # Skema tidak terlalu penting di sini
    mock_get_schema.return_value = mock_schema_sql
    table_name = "empty_tuples_test"
    caplog.set_level(logging.INFO) # Tangkap log INFO spesifik

    # --- Konfigurasi patch kunci ---
    # Mock ini akan dipanggil ketika df_prepared.to_numpy() dieksekusi
    mock_to_numpy.return_value = []

    # Act
    result = load.load_to_postgres(sample_df, mock_db_config, table_name)

    # Assert
    assert result is True # Harus mengembalikan True sesuai baris 247

    # Pemeriksaan dasar pengaturan koneksi
    mock_connect.assert_called_once_with(**mock_db_config)
    mock_get_schema.assert_called_once_with(sample_df)

    # Periksa CREATE/TRUNCATE tetap terjadi
    mock_cursor.execute.assert_has_calls([
        call(sql.SQL('CREATE TABLE IF NOT EXISTS {table} ({schema});').format(
            table=sql.Identifier(table_name), schema=mock_schema_sql
        )),
        call(sql.SQL('TRUNCATE TABLE {table};').format(table=sql.Identifier(table_name))),
    ], any_order=False)

    # Verifikasi patch kita dipanggil (pada df_prepared)
    # Menggunakan assert_called() untuk fleksibilitas jika dipanggil > 1 kali oleh internal pandas
    mock_to_numpy.assert_called()
    # Jika Anda yakin hanya dipanggil sekali untuk tujuan ini:
    # mock_to_numpy.assert_called_once()

    # --- Periksa Baris 244-246 ---
    # 1. Periksa pesan log spesifik
    expected_log = "No valid data tuples to insert into PostgreSQL."
    assert expected_log in caplog.text
    # 2. Periksa commit dipanggil *di dalam blok ini*
    mock_conn.commit.assert_called_once() # Baris 246
    # --- Akhir Pemeriksaan ---

    # Pastikan logika penyisipan utama dilewati
    mock_execute_values.assert_not_called()

    # Pastikan rollback tidak dipanggil
    mock_conn.rollback.assert_not_called()

    # Pastikan koneksi ditutup di blok finally
    mock_conn.close.assert_called_once()


@patch("utils.load.psycopg2.connect")
@patch("utils.load.execute_values")
def test_load_to_postgres_empty_df(mock_execute_values, mock_connect, empty_df, mock_db_config, caplog):
    """Tests skipping PostgreSQL load for an empty DataFrame."""
    # Arrange
    table_name = "empty_test"
    caplog.set_level(logging.WARNING)

    # Act
    result = load.load_to_postgres(empty_df, mock_db_config, table_name)

    # Assert
    assert result is True
    mock_connect.assert_not_called() # Should not even connect if df is empty
    mock_execute_values.assert_not_called()
    assert f"DataFrame is empty. Skipping PostgreSQL load to table '{table_name}'." in caplog.text


@patch("utils.load.psycopg2.connect", side_effect=Psycopg2Error("Connection failed"))
def test_load_to_postgres_connection_error(mock_connect, sample_df, mock_db_config, caplog):
    """Tests handling of connection errors."""
    # Arrange
    table_name = "conn_error_test"
    caplog.set_level(logging.ERROR)

    # Act
    result = load.load_to_postgres(sample_df, mock_db_config, table_name)

    # Assert
    assert result is False
    mock_connect.assert_called_once_with(**mock_db_config)
    assert f"PostgreSQL error during load to table '{table_name}': Connection failed" in caplog.text


@patch("utils.load.psycopg2.connect")
@patch("utils.load._get_postgres_schema") # Need to mock schema generation
def test_load_to_postgres_execute_error(mock_get_schema, mock_connect, sample_df, mock_db_config, caplog):
    """Tests handling of errors during cursor execution (e.g., TRUNCATE)."""
    # Arrange
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_conn.cursor.return_value.__exit__.return_value = None # Handle exit context manager
    # Simulate schema generation
    mock_schema_sql = sql.SQL('"col1" TEXT')
    mock_get_schema.return_value = mock_schema_sql
    mock_cursor.execute.side_effect = [
        None, # CREATE TABLE succeeds
        Psycopg2Error("Truncate failed") # TRUNCATE fails
    ]
    table_name = "exec_error_test"
    caplog.set_level(logging.ERROR)

    # Act
    result = load.load_to_postgres(sample_df, mock_db_config, table_name)

    # Assert
    assert result is False
    mock_get_schema.assert_called_once_with(sample_df)
    assert mock_cursor.execute.call_count == 2 # Called for CREATE and TRUNCATE
    mock_conn.commit.assert_not_called()
    mock_conn.rollback.assert_called_once() # Transaction should be rolled back
    mock_conn.close.assert_called_once()
    assert f"PostgreSQL error during load to table '{table_name}': Truncate failed" in caplog.text


@patch("utils.load.psycopg2.connect")
@patch("utils.load._get_postgres_schema") # Need to mock schema generation
@patch("utils.load.execute_values", side_effect=Psycopg2Error("Insert failed"))
def test_load_to_postgres_execute_values_error(mock_execute_values, mock_get_schema, mock_connect, sample_df, mock_db_config, caplog):
    """Tests handling of errors during bulk insert."""
    # Arrange
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_conn.cursor.return_value.__exit__.return_value = None # Handle exit context manager
    mock_schema_sql = sql.SQL('"col1" TEXT')
    mock_get_schema.return_value = mock_schema_sql
    table_name = "insert_error_test"
    caplog.set_level(logging.ERROR)

    # Act
    result = load.load_to_postgres(sample_df, mock_db_config, table_name)

    # Assert
    assert result is False
    mock_get_schema.assert_called_once_with(sample_df)
    mock_execute_values.assert_called_once()
    mock_conn.commit.assert_not_called()
    mock_conn.rollback.assert_called_once()
    mock_conn.close.assert_called_once()
    assert f"PostgreSQL error during load to table '{table_name}': Insert failed" in caplog.text


@patch("utils.load.psycopg2.connect")
def test_load_to_postgres_missing_db_config_key(mock_connect, sample_df, mock_db_config, caplog):
    """Tests handling of missing keys in the db_config dictionary."""
    # Arrange
    incomplete_config = mock_db_config.copy()
    missing_key = "password"
    del incomplete_config[missing_key] # Remove a required key
    table_name = "config_error_test"
    caplog.set_level(logging.ERROR)

    # Set up the mock to raise KeyError when called with the incomplete config
    def connect_side_effect(*args, **kwargs):
        if missing_key not in kwargs:
            raise KeyError(missing_key)
        return MagicMock() # Should not be reached in this test
    mock_connect.side_effect = connect_side_effect

    # Act
    result = load.load_to_postgres(sample_df, incomplete_config, table_name)

    # Assert
    assert result is False
    mock_connect.assert_called_once_with(**incomplete_config) # Connection attempt is made
    assert f"Missing key in db_config: '{missing_key}'" in caplog.text


@patch("utils.load.psycopg2.connect")
@patch("utils.load._get_postgres_schema")
@patch("utils.load.execute_values")
def test_load_to_postgres_unexpected_error_during_commit(mock_execute_values, mock_get_schema, mock_connect, sample_df, mock_db_config, caplog):
    """Tests handling of unexpected non-DB errors during commit."""
    # Arrange
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_conn.cursor.return_value.__exit__.return_value = None # Handle exit context manager
    mock_schema_sql = sql.SQL('"col1" TEXT')
    mock_get_schema.return_value = mock_schema_sql
    # Simulate error during commit
    mock_conn.commit.side_effect = Exception("Unexpected commit error")
    table_name = "unexpected_commit_err_test"
    caplog.set_level(logging.ERROR)

    # Act
    result = load.load_to_postgres(sample_df, mock_db_config, table_name)

    # Assert
    assert result is False
    mock_connect.assert_called_once()
    mock_get_schema.assert_called_once()
    mock_execute_values.assert_called_once() # Should be called before commit fails
    mock_conn.commit.assert_called_once() # Error happens during commit
    mock_conn.rollback.assert_called_once() # Rollback *should* be called because conn exists
    mock_conn.close.assert_called_once() # Close should be called in finally
    assert f"An unexpected error occurred during PostgreSQL load to '{table_name}'" in caplog.text
    assert "Unexpected commit error" in caplog.text
    assert "PostgreSQL transaction rolled back due to unexpected error." in caplog.text # Check log message


@patch("utils.load.psycopg2.connect")
@patch("utils.load._get_postgres_schema")
@patch("utils.load.execute_values", side_effect=Exception("Unexpected generic error during execute_values"))
def test_load_to_postgres_unexpected_error_during_execute_values(mock_execute_values, mock_get_schema, mock_connect, sample_df, mock_db_config, caplog):
    """Tests handling of unexpected non-DB errors ensuring rollback is covered."""
    # Arrange
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_conn.cursor.return_value.__exit__.return_value = None # Handle exit context manager
    mock_schema_sql = sql.SQL('"col1" TEXT') # Dummy schema
    mock_get_schema.return_value = mock_schema_sql
    table_name = "unexpected_generic_err_test"
    caplog.set_level(logging.ERROR)

    # Act
    result = load.load_to_postgres(sample_df, mock_db_config, table_name)

    # Assert
    assert result is False
    mock_connect.assert_called_once_with(**mock_db_config)
    mock_get_schema.assert_called_once_with(sample_df)
    # Check that execute was called for CREATE/TRUNCATE before the error
    assert mock_cursor.execute.call_count >= 2
    mock_execute_values.assert_called_once() # The call that raises the error

    # Crucially, assert rollback was called because 'conn' exists when Exception is caught
    mock_conn.rollback.assert_called_once()
    mock_conn.commit.assert_not_called() # Commit should not be reached
    mock_conn.close.assert_called_once() # Finally block should still close connection

    assert f"An unexpected error occurred during PostgreSQL load to '{table_name}'" in caplog.text
    assert "Unexpected generic error during execute_values" in caplog.text
    # Check the specific log message from the 'if conn:' block
    assert "PostgreSQL transaction rolled back due to unexpected error." in caplog.text