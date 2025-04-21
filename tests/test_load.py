# tests/test_load.py
"""
Unit tests for the utils.load module.
"""

import logging
import os
from unittest.mock import MagicMock, call, patch

import gspread
import pandas as pd
import psycopg2
import pytest
from gspread.exceptions import GSpreadException, WorksheetNotFound
from psycopg2 import Error as Psycopg2Error
from psycopg2 import sql

# Assuming utils is importable from the project root
from utils import load
from utils.constants import WORKSHEET_NAME

# --- Sample Data ---
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
    return pd.DataFrame(data)

@pytest.fixture
def empty_df():
    """Provides an empty DataFrame."""
    return pd.DataFrame()

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


@patch("pandas.DataFrame.to_csv")
def test_load_to_csv_empty_df(mock_to_csv, empty_df, tmp_path, caplog):
    """Tests skipping save for an empty DataFrame."""
    # Arrange
    filepath = tmp_path / "empty.csv"
    caplog.set_level(logging.WARNING)

    # Act
    result = load.load_to_csv(empty_df, str(filepath))

    # Assert
    assert result is True
    mock_to_csv.assert_not_called()
    assert f"DataFrame is empty. Skipping CSV save to {filepath}" in caplog.text


@patch("pandas.DataFrame.to_csv", side_effect=IOError("Disk full"))
def test_load_to_csv_io_error(mock_to_csv, sample_df, tmp_path, caplog):
    """Tests handling of IOError during CSV save."""
    # Arrange
    filepath = tmp_path / "error.csv"
    caplog.set_level(logging.ERROR)

    # Act
    result = load.load_to_csv(sample_df, str(filepath))

    # Assert
    assert result is False
    mock_to_csv.assert_called_once()
    assert f"Error saving data to CSV file {filepath}: Disk full" in caplog.text


@patch("pandas.DataFrame.to_csv", side_effect=Exception("Unexpected pandas error"))
def test_load_to_csv_unexpected_error(mock_to_csv, sample_df, tmp_path, caplog):
    """Tests handling of unexpected exceptions during CSV save."""
    # Arrange
    filepath = tmp_path / "unexpected.csv"
    caplog.set_level(logging.ERROR)

    # Act
    result = load.load_to_csv(sample_df, str(filepath))

    # Assert
    assert result is False
    mock_to_csv.assert_called_once()
    assert f"An unexpected error occurred during CSV saving to {filepath}" in caplog.text
    assert "Unexpected pandas error" in caplog.text

# --- Tests for load_to_gsheets ---

@patch("utils.load.gspread.authorize")
@patch("utils.load.Credentials.from_service_account_file")
def test_load_to_gsheets_success_worksheet_exists(mock_creds, mock_authorize, sample_df):
    """Tests successful load to an existing Google Sheet worksheet."""
    # Arrange
    mock_gc = MagicMock()
    mock_spreadsheet = MagicMock()
    mock_worksheet = MagicMock()
    mock_authorize.return_value = mock_gc
    mock_gc.open_by_key.return_value = mock_spreadsheet
    mock_spreadsheet.worksheet.return_value = mock_worksheet

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
    mock_spreadsheet.worksheet.assert_called_once_with(worksheet_name)
    mock_spreadsheet.add_worksheet.assert_not_called()
    mock_worksheet.clear.assert_called_once()
    # Check that update is called with header + data rows
    assert mock_worksheet.update.call_count == 1
    update_args, update_kwargs = mock_worksheet.update.call_args
    assert len(update_args[0]) == len(sample_df) + 1 # Header + data
    assert update_args[0][0] == list(sample_df.columns) # Check header row
    assert update_args[0][1][0] == "Product A" # Check first data cell
    # Check timestamp formatting
    assert update_args[0][1][-1] == "2024-01-15 10:00:00+0700"
    assert update_kwargs == {"value_input_option": "USER_ENTERED"}


@patch("utils.load.gspread.authorize")
@patch("utils.load.Credentials.from_service_account_file")
def test_load_to_gsheets_success_create_worksheet(mock_creds, mock_authorize, sample_df):
    """Tests successful load when the worksheet needs to be created."""
    # Arrange
    mock_gc = MagicMock()
    mock_spreadsheet = MagicMock()
    mock_new_worksheet = MagicMock()
    mock_authorize.return_value = mock_gc
    mock_gc.open_by_key.return_value = mock_spreadsheet
    mock_spreadsheet.worksheet.side_effect = WorksheetNotFound
    mock_spreadsheet.add_worksheet.return_value = mock_new_worksheet

    credentials_path = "fake_creds.json"
    sheet_id = "fake_sheet_id"
    worksheet_name = "NewSheet"

    # Act
    result = load.load_to_gsheets(sample_df, credentials_path, sheet_id, worksheet_name)

    # Assert
    assert result is True
    mock_spreadsheet.worksheet.assert_called_once_with(worksheet_name)
    mock_spreadsheet.add_worksheet.assert_called_once_with(title=worksheet_name, rows=1, cols=1)
    mock_new_worksheet.clear.assert_called_once()
    mock_new_worksheet.update.assert_called_once() # Data loaded into the new sheet


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
    assert "DataFrame is empty. Skipping Google Sheets load" in caplog.text


@patch("utils.load.Credentials.from_service_account_file", side_effect=FileNotFoundError("Creds not found"))
def test_load_to_gsheets_credentials_not_found(mock_creds, sample_df, caplog):
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
    mock_authorize.assert_called_once()
    assert "Google Sheets API or client error: API Error" in caplog.text


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

@pytest.fixture
def mock_db_config():
    return {"host": "localhost", "port": "5432", "dbname": "testdb", "user": "testuser", "password": "password"}

@patch("utils.load.psycopg2.connect")
@patch("utils.load._get_postgres_schema")
@patch("utils.load.execute_values")
def test_load_to_postgres_success(mock_execute_values, mock_get_schema, mock_connect, sample_df, mock_db_config):
    """Tests successful loading data into PostgreSQL."""
    # Arrange
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    # Simulate schema generation
    mock_schema_sql = sql.SQL('"col1" TEXT, "col2" INTEGER')
    mock_get_schema.return_value = mock_schema_sql
    table_name = "products_test"

    # Act
    result = load.load_to_postgres(sample_df, mock_db_config, table_name)

    # Assert
    assert result is True
    mock_connect.assert_called_once_with(**mock_db_config)
    mock_get_schema.assert_called_once_with(sample_df)
    assert mock_conn.autocommit is False # Check transaction usage

    # Check SQL execution calls
    expected_calls = [
        # CREATE TABLE IF NOT EXISTS "products_test" ("col1" TEXT, "col2" INTEGER);
        call(sql.SQL('CREATE TABLE IF NOT EXISTS {table} ({schema});').format(
            table=sql.Identifier(table_name), schema=mock_schema_sql
        )),
        # TRUNCATE TABLE "products_test";
        call(sql.SQL('TRUNCATE TABLE {table};').format(table=sql.Identifier(table_name))),
    ]
    mock_cursor.execute.assert_has_calls(expected_calls)

    # Check execute_values call
    assert mock_execute_values.call_count == 1
    call_args, call_kwargs = mock_execute_values.call_args
    # call_args[0] is the cursor
    # call_args[1] is the insert query string
    # call_args[2] is the list of tuples (data)
    assert f'INSERT INTO "{table_name}"' in call_args[1]
    assert len(call_args[2]) == len(sample_df) # Check number of data rows
    assert isinstance(call_args[2][0], tuple) # Check data format

    mock_conn.commit.assert_called_once()
    mock_conn.rollback.assert_not_called()
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
def test_load_to_postgres_execute_error(mock_connect, sample_df, mock_db_config, caplog):
    """Tests handling of errors during cursor execution (e.g., TRUNCATE)."""
    # Arrange
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
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
    assert mock_cursor.execute.call_count == 2 # Called for CREATE and TRUNCATE
    mock_conn.commit.assert_not_called()
    mock_conn.rollback.assert_called_once() # Transaction should be rolled back
    mock_conn.close.assert_called_once()
    assert f"PostgreSQL error during load to table '{table_name}': Truncate failed" in caplog.text


@patch("utils.load.psycopg2.connect")
@patch("utils.load.execute_values", side_effect=Psycopg2Error("Insert failed"))
def test_load_to_postgres_execute_values_error(mock_execute_values, mock_connect, sample_df, mock_db_config, caplog):
    """Tests handling of errors during bulk insert."""
    # Arrange
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    table_name = "insert_error_test"
    caplog.set_level(logging.ERROR)

    # Act
    result = load.load_to_postgres(sample_df, mock_db_config, table_name)

    # Assert
    assert result is False
    mock_execute_values.assert_called_once()
    mock_conn.commit.assert_not_called()
    mock_conn.rollback.assert_called_once()
    mock_conn.close.assert_called_once()
    assert f"PostgreSQL error during load to table '{table_name}': Insert failed" in caplog.text


def test_load_to_postgres_missing_db_config_key(sample_df, mock_db_config, caplog):
    """Tests handling of missing keys in the db_config dictionary."""
    # Arrange
    incomplete_config = mock_db_config.copy()
    del incomplete_config["password"] # Remove a required key
    table_name = "config_error_test"
    caplog.set_level(logging.ERROR)

    # Act
    result = load.load_to_postgres(sample_df, incomplete_config, table_name)

    # Assert
    assert result is False
    # Connection attempt might raise KeyError before psycopg2 does
    assert "Missing key in db_config: 'password'" in caplog.text


@patch("utils.load.psycopg2.connect")
def test_load_to_postgres_unexpected_error(mock_connect, sample_df, mock_db_config, caplog):
    """Tests handling of unexpected errors during PostgreSQL operations."""
    # Arrange
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    # Simulate error during commit
    mock_conn.commit.side_effect = Exception("Unexpected commit error")
    table_name = "unexpected_err_test"
    caplog.set_level(logging.ERROR)

    # Act
    result = load.load_to_postgres(sample_df, mock_db_config, table_name)

    # Assert
    assert result is False
    mock_conn.commit.assert_called_once() # Error happens during commit
    mock_conn.rollback.assert_called_once() # Rollback should still be called in finally
    mock_conn.close.assert_called_once()
    assert f"An unexpected error occurred during PostgreSQL load to '{table_name}'" in caplog.text
    assert "Unexpected commit error" in caplog.text