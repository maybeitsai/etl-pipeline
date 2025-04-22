# tests/test_load.py
"""
Unit tests for the utils.load module.
"""

import logging
import os
from unittest.mock import MagicMock, call, patch

import gspread
import numpy as np
import pandas as pd
import psycopg2
import pytest
from gspread.exceptions import GSpreadException, WorksheetNotFound
from psycopg2 import Error as Psycopg2Error
from psycopg2 import sql

# Import Composed specifically for patching its method if needed
from psycopg2.sql import Composed
from psycopg2.extras import (
    execute_values,
)

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
            ["2024-01-15 10:00:00+07:00", "2024-01-15 11:00:00+07:00"], utc=True
        ).tz_convert("Asia/Jakarta"),
    }
    df = pd.DataFrame(data).astype(
        {
            "title": str,
            "price": float,
            "rating": float,
            "colors": "Int64",
            "size": str,
            "gender": str,
            "image_url": str,
        }
    )
    return df


@pytest.fixture
def empty_df():
    return pd.DataFrame()


@pytest.fixture
def mock_db_config():
    return {
        "host": "localhost",
        "port": "5432",
        "dbname": "testdb",
        "user": "testuser",
        "password": "password",
    }


# --- Tests for load_to_csv ---
@patch("utils.load.os.makedirs")
@patch("pandas.DataFrame.to_csv")
def test_load_to_csv_success(mock_to_csv, mock_makedirs, sample_df, tmp_path):
    filepath = tmp_path / "output" / "data.csv"
    result = load.load_to_csv(sample_df, str(filepath))
    assert result is True
    mock_makedirs.assert_called_once_with(str(tmp_path / "output"), exist_ok=True)
    mock_to_csv.assert_called_once_with(str(filepath), index=False, encoding="utf-8")


@patch("utils.load.os.makedirs")
@patch("pandas.DataFrame.to_csv")
def test_load_to_csv_no_subdir(mock_to_csv, mock_makedirs, sample_df, tmp_path):
    filepath = tmp_path / "data.csv"
    result = load.load_to_csv(sample_df, str(filepath))
    assert result is True
    mock_makedirs.assert_called_once_with(str(tmp_path), exist_ok=True)
    mock_to_csv.assert_called_once_with(str(filepath), index=False, encoding="utf-8")


@patch("utils.load.os.makedirs")
@patch("pandas.DataFrame.to_csv")
def test_load_to_csv_empty_df(mock_to_csv, mock_makedirs, empty_df, tmp_path, caplog):
    filepath = tmp_path / "empty.csv"
    caplog.set_level(logging.WARNING)
    result = load.load_to_csv(empty_df, str(filepath))
    assert result is True
    mock_makedirs.assert_not_called()
    mock_to_csv.assert_not_called()
    assert f"DataFrame is empty. Skipping CSV save to {filepath}" in caplog.text


@patch("utils.load.os.makedirs")
@patch("pandas.DataFrame.to_csv", side_effect=IOError("Disk full"))
def test_load_to_csv_io_error(mock_to_csv, mock_makedirs, sample_df, tmp_path, caplog):
    filepath = tmp_path / "error.csv"
    caplog.set_level(logging.ERROR)
    result = load.load_to_csv(sample_df, str(filepath))
    assert result is False
    mock_makedirs.assert_called_once()
    mock_to_csv.assert_called_once()
    assert f"Error saving data to CSV file {filepath}: Disk full" in caplog.text


@patch("utils.load.os.makedirs")
@patch("pandas.DataFrame.to_csv", side_effect=OSError("Permission denied"))
def test_load_to_csv_os_error(mock_to_csv, mock_makedirs, sample_df, tmp_path, caplog):
    filepath = tmp_path / "os_error.csv"
    caplog.set_level(logging.ERROR)
    result = load.load_to_csv(sample_df, str(filepath))
    assert result is False
    mock_makedirs.assert_called_once()
    mock_to_csv.assert_called_once()
    assert f"Error saving data to CSV file {filepath}: Permission denied" in caplog.text


@patch("utils.load.os.makedirs")
@patch("pandas.DataFrame.to_csv", side_effect=Exception("Unexpected pandas error"))
def test_load_to_csv_unexpected_error(
    mock_to_csv, mock_makedirs, sample_df, tmp_path, caplog
):
    filepath = tmp_path / "unexpected.csv"
    caplog.set_level(logging.ERROR)
    result = load.load_to_csv(sample_df, str(filepath))
    assert result is False
    mock_makedirs.assert_called_once()
    mock_to_csv.assert_called_once()
    assert (
        f"An unexpected error occurred during CSV saving to {filepath}" in caplog.text
    )
    assert "Unexpected pandas error" in caplog.text


# --- Tests for _prepare_gsheets_data ---
def test_prepare_gsheets_data_success(sample_df):
    df = sample_df.copy()
    df.loc[0, "rating"] = np.nan
    df.loc[1, "timestamp"] = pd.NaT
    df.loc[0, "colors"] = pd.NA
    result_list = load._prepare_gsheets_data(df)
    assert result_list[0] == list(df.columns)
    assert result_list[1][-1] == "2024-01-15 10:00:00+0700"
    assert result_list[2][-1] is None


def test_prepare_gsheets_data_no_timestamp(sample_df):
    df_no_ts = sample_df.drop(columns=["timestamp"])
    result_list = load._prepare_gsheets_data(df_no_ts)
    assert list(df_no_ts.columns) == result_list[0]


def test_prepare_gsheets_data_timestamp_not_datetime(sample_df):
    df_wrong_ts = sample_df.copy()
    df_wrong_ts["timestamp"] = ["today", "yesterday"]
    result_list = load._prepare_gsheets_data(df_wrong_ts)
    assert result_list[1][-1] == "today"


# --- Tests for load_to_gsheets ---
@patch("utils.load._prepare_gsheets_data")
@patch("utils.load.gspread.authorize")
@patch("utils.load.Credentials.from_service_account_file")
def test_load_to_gsheets_success_worksheet_exists(
    mock_creds, mock_authorize, mock_prepare, sample_df
):
    mock_gc = MagicMock()
    mock_spreadsheet = MagicMock()
    mock_worksheet = MagicMock()
    mock_authorize.return_value = mock_gc
    mock_gc.open_by_key.return_value = mock_spreadsheet
    mock_spreadsheet.worksheet.return_value = mock_worksheet
    prepared_data = [["col1", "col2"], ["a", 1], ["b", 2]]
    mock_prepare.return_value = prepared_data
    credentials_path = "fake_creds.json"
    sheet_id = "fake_sheet_id"
    worksheet_name = "TestData"
    result = load.load_to_gsheets(sample_df, credentials_path, sheet_id, worksheet_name)
    assert result is True
    mock_creds.assert_called_once()
    mock_authorize.assert_called_once()
    mock_gc.open_by_key.assert_called_once_with(sheet_id)
    mock_prepare.assert_called_once_with(sample_df)
    mock_spreadsheet.worksheet.assert_called_once_with(worksheet_name)
    mock_worksheet.clear.assert_called_once()
    mock_worksheet.update.assert_called_once_with(
        prepared_data, value_input_option="USER_ENTERED"
    )


@patch("utils.load._prepare_gsheets_data")
@patch("utils.load.gspread.authorize")
@patch("utils.load.Credentials.from_service_account_file")
def test_load_to_gsheets_success_create_worksheet(
    mock_creds, mock_authorize, mock_prepare, sample_df
):
    mock_gc = MagicMock()
    mock_spreadsheet = MagicMock()
    mock_new_worksheet = MagicMock()
    mock_authorize.return_value = mock_gc
    mock_gc.open_by_key.return_value = mock_spreadsheet
    mock_spreadsheet.worksheet.side_effect = WorksheetNotFound
    mock_spreadsheet.add_worksheet.return_value = mock_new_worksheet
    prepared_data = [["col1"], ["a"]]
    mock_prepare.return_value = prepared_data
    credentials_path = "fake_creds.json"
    sheet_id = "fake_sheet_id"
    worksheet_name = "NewSheet"
    result = load.load_to_gsheets(sample_df, credentials_path, sheet_id, worksheet_name)
    assert result is True
    mock_prepare.assert_called_once_with(sample_df)
    mock_spreadsheet.worksheet.assert_called_once_with(worksheet_name)
    mock_spreadsheet.add_worksheet.assert_called_once_with(
        title=worksheet_name, rows=1, cols=1
    )
    mock_new_worksheet.clear.assert_called_once()
    mock_new_worksheet.update.assert_called_once_with(
        prepared_data, value_input_option="USER_ENTERED"
    )


@patch("utils.load.gspread.authorize")
@patch("utils.load.Credentials.from_service_account_file")
def test_load_to_gsheets_empty_df(mock_creds, mock_authorize, empty_df, caplog):
    credentials_path = "fake_creds.json"
    sheet_id = "fake_sheet_id"
    worksheet_name = "EmptySheet"
    caplog.set_level(logging.WARNING)
    result = load.load_to_gsheets(empty_df, credentials_path, sheet_id, worksheet_name)
    assert result is True
    mock_creds.assert_not_called()
    assert f"DataFrame is empty. Skipping Google Sheets load" in caplog.text


@patch("utils.load.gspread.authorize")
@patch(
    "utils.load.Credentials.from_service_account_file",
    side_effect=FileNotFoundError("Creds not found"),
)
def test_load_to_gsheets_credentials_not_found(
    mock_creds, mock_authorize, sample_df, caplog
):
    credentials_path = "nonexistent_creds.json"
    sheet_id = "fake_sheet_id"
    worksheet_name = "TestData"
    caplog.set_level(logging.ERROR)
    result = load.load_to_gsheets(sample_df, credentials_path, sheet_id, worksheet_name)
    assert result is False
    mock_creds.assert_called_once()
    assert f"Credentials file not found at: {credentials_path}" in caplog.text


@patch("utils.load.gspread.authorize", side_effect=GSpreadException("API Error"))
@patch("utils.load.Credentials.from_service_account_file")
def test_load_to_gsheets_gspread_error(mock_creds, mock_authorize, sample_df, caplog):
    credentials_path = "fake_creds.json"
    sheet_id = "fake_sheet_id"
    worksheet_name = "TestData"
    caplog.set_level(logging.ERROR)
    result = load.load_to_gsheets(sample_df, credentials_path, sheet_id, worksheet_name)
    assert result is False
    mock_creds.assert_called_once()
    assert "Google Sheets API or client error: API Error" in caplog.text


@patch("utils.load.gspread.authorize")
@patch("utils.load.Credentials.from_service_account_file")
@patch("utils.load._prepare_gsheets_data", side_effect=ValueError("Bad data format"))
def test_load_to_gsheets_value_error(
    mock_prepare_data, mock_creds, mock_authorize, sample_df, caplog
):
    mock_gc = MagicMock()
    mock_authorize.return_value = mock_gc
    credentials_path = "fake_creds.json"
    sheet_id = "fake_sheet_id"
    worksheet_name = "TestData"
    caplog.set_level(logging.ERROR)
    result = load.load_to_gsheets(sample_df, credentials_path, sheet_id, worksheet_name)
    assert result is False
    mock_prepare_data.assert_called_once()
    assert "Value error during Google Sheets operation: Bad data format" in caplog.text


@patch("utils.load.gspread.authorize")
@patch("utils.load.Credentials.from_service_account_file")
def test_load_to_gsheets_unexpected_error(
    mock_creds, mock_authorize, sample_df, caplog
):
    mock_gc = MagicMock()
    mock_authorize.return_value = mock_gc
    mock_gc.open_by_key.side_effect = Exception("Something else went wrong")
    credentials_path = "fake_creds.json"
    sheet_id = "fake_sheet_id"
    worksheet_name = "TestData"
    caplog.set_level(logging.ERROR)
    result = load.load_to_gsheets(sample_df, credentials_path, sheet_id, worksheet_name)
    assert result is False
    assert "An unexpected error occurred during Google Sheets loading" in caplog.text


# --- Tests for _get_postgres_schema ---
def test_get_postgres_schema_standard_types():
    """Tests schema generation for standard pandas dtypes."""
    df = pd.DataFrame(
        {
            "col_str": ["a", "b"],
            "col_int": [1, 2],
            "col_float": [1.1, 2.2],
            "col_bool": [True, False],
            "col_dt_naive": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "price": [10.5, 20.0],
        }
    ).astype(
        {
            "col_str": "object",
            "col_int": "int64",
            "col_float": "float64",
            "col_bool": "bool",
            "price": "float64",
            "col_dt_naive": "datetime64[ns]",
        }
    )

    # Mock the SQL composition
    with patch("psycopg2.sql.SQL", side_effect=lambda x: x), patch(
        "psycopg2.sql.Identifier", side_effect=lambda x: f'"{x}"'
    ), patch("psycopg2.sql.Composed", side_effect=lambda x: ", ".join(x)):

        schema_sql = load._get_postgres_schema(df)

        # Now check the full schema string content
        assert '"col_str" TEXT' in schema_sql
        assert '"col_int" INTEGER' in schema_sql
        assert '"col_float" REAL' in schema_sql
        assert '"col_bool" BOOLEAN' in schema_sql
        assert '"col_dt_naive" TIMESTAMP' in schema_sql
        assert '"price" NUMERIC(12, 2)' in schema_sql


def test_get_postgres_schema_datetime_aware():
    """Tests schema generation for timezone-aware datetime."""
    df = pd.DataFrame(
        {
            "col_dt_aware": pd.to_datetime(
                ["2024-01-01 10:00:00+07:00"], utc=True
            ).tz_convert("Asia/Jakarta")
        }
    )

    # Mock the SQL composition
    with patch("psycopg2.sql.SQL", side_effect=lambda x: x), patch(
        "psycopg2.sql.Identifier", side_effect=lambda x: f'"{x}"'
    ), patch("psycopg2.sql.Composed", side_effect=lambda x: ", ".join(x)):

        schema_sql = load._get_postgres_schema(df)
        assert '"col_dt_aware" TIMESTAMPTZ' in schema_sql


def test_get_postgres_schema_unmapped_type(caplog):
    """Tests schema generation fallback for unmapped dtypes."""
    df = pd.DataFrame({"col_complex": [1 + 2j, 3 + 4j]})
    caplog.set_level(logging.WARNING)

    # Mock the SQL composition
    with patch("psycopg2.sql.SQL", side_effect=lambda x: x), patch(
        "psycopg2.sql.Identifier", side_effect=lambda x: f'"{x}"'
    ), patch("psycopg2.sql.Composed", side_effect=lambda x: ", ".join(x)):

        schema_sql = load._get_postgres_schema(df)
        assert '"col_complex" TEXT' in schema_sql
        assert (
            "Unmapped dtype 'complex128' for column 'col_complex'. Defaulting to TEXT."
            in caplog.text
        )


# Helper function for mock setup
def setup_mock_conn_cursor(mock_connect):
    """Sets up mock connection and cursor with necessary attributes."""
    mock_conn = MagicMock(name="MockConnection")
    mock_cursor = MagicMock(name="MockCursor")
    mock_connect.return_value = mock_conn
    mock_conn.encoding = "utf-8"
    mock_cursor.connection = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_conn.cursor.return_value.__exit__.return_value = None
    mock_cursor.execute.return_value = None
    return mock_conn, mock_cursor


# --- Tests for load_to_postgres ---
@patch("utils.load.psycopg2.connect")
@patch("utils.load._get_postgres_schema")
@patch("utils.load.execute_values")
def test_load_to_postgres_success_covers_final_commit_log_return(
    mock_execute_values,
    mock_get_schema,
    mock_connect,
    sample_df,
    mock_db_config,
    caplog,
):
    """Tests successful load, focusing on final commit, log, and return."""
    # Arrange
    mock_conn, mock_cursor = setup_mock_conn_cursor(mock_connect)
    mock_schema_sql = "col1 TEXT, col2 INT"  # Simple string instead of mock
    mock_get_schema.return_value = mock_schema_sql
    table_name = "products_final_check"
    caplog.set_level(logging.INFO)

    df_prepared_expected = sample_df.astype(object).where(pd.notnull(sample_df), None)
    expected_data_tuples = [tuple(x) for x in df_prepared_expected.to_numpy()]
    expected_row_count = len(expected_data_tuples)

    # Important: Create a proper mock for SQL.as_string() to pass to execute_values
    mock_sql_obj = MagicMock()
    mock_sql_obj.as_string.return_value = (
        f"INSERT INTO {table_name} (col1, col2) VALUES %s"
    )

    # Mock the sql.SQL to return our prepared mock
    with patch("psycopg2.sql.SQL", return_value=mock_sql_obj), patch(
        "psycopg2.sql.Identifier", side_effect=lambda x: x
    ):

        # Act
        result = load.load_to_postgres(sample_df, mock_db_config, table_name)

    # Assert
    assert (
        result is True
    ), f"Function should return True on success. Logs: {caplog.text}"
    mock_connect.assert_called_once_with(**mock_db_config)
    mock_get_schema.assert_called_once_with(sample_df)

    # Check execute_values was called
    mock_execute_values.assert_called_once()
    args = mock_execute_values.call_args[0]
    assert args[0] is mock_cursor
    assert len(args) >= 3

    # Check if the third argument (data tuples) is correct
    actual_data = args[2]
    assert len(actual_data) == len(
        expected_data_tuples
    ), f"Expected {len(expected_data_tuples)} data tuples, got {len(actual_data)}"

    mock_conn.commit.assert_called_once()
    assert (
        "Successfully loaded" in caplog.text
        and f"into PostgreSQL table '{table_name}'" in caplog.text
    )
    mock_conn.rollback.assert_not_called()
    mock_conn.close.assert_called_once()


@patch("utils.load.psycopg2.connect")
@patch("utils.load._get_postgres_schema")
@patch("pandas.DataFrame.to_numpy")
@patch("utils.load.execute_values")
def test_load_to_postgres_force_empty_tuples_path(
    mock_execute_values,
    mock_to_numpy,
    mock_get_schema,
    mock_connect,
    sample_df,
    mock_db_config,
    caplog,
):
    """Tests the specific path where data_tuples becomes empty after processing."""
    assert not sample_df.empty
    mock_conn, mock_cursor = setup_mock_conn_cursor(mock_connect)
    mock_schema_sql = sql.SQL('"dummy" TEXT')
    mock_get_schema.return_value = mock_schema_sql
    table_name = "empty_tuples_test"
    caplog.set_level(logging.INFO)
    mock_to_numpy.return_value = []
    result = load.load_to_postgres(sample_df, mock_db_config, table_name)
    assert result is True
    mock_connect.assert_called_once_with(**mock_db_config)
    mock_get_schema.assert_called_once_with(sample_df)
    calls = mock_cursor.execute.call_args_list
    assert any("CREATE TABLE IF NOT EXISTS" in str(c.args[0]) for c in calls)
    assert any("TRUNCATE TABLE" in str(c.args[0]) for c in calls)
    mock_to_numpy.assert_called()
    assert "No valid data tuples to insert into PostgreSQL." in caplog.text
    mock_conn.commit.assert_called_once()
    mock_execute_values.assert_not_called()
    mock_conn.rollback.assert_not_called()
    mock_conn.close.assert_called_once()


@patch("utils.load.psycopg2.connect")
@patch("utils.load.execute_values")
def test_load_to_postgres_empty_df(
    mock_execute_values, mock_connect, empty_df, mock_db_config, caplog
):
    """Tests skipping PostgreSQL load for an empty DataFrame."""
    table_name = "empty_test"
    caplog.set_level(logging.WARNING)
    result = load.load_to_postgres(empty_df, mock_db_config, table_name)
    assert result is True
    mock_connect.assert_not_called()
    mock_execute_values.assert_not_called()
    assert (
        f"DataFrame is empty. Skipping PostgreSQL load to table '{table_name}'."
        in caplog.text
    )


@patch("utils.load.psycopg2.connect", side_effect=Psycopg2Error("Connection failed"))
def test_load_to_postgres_connection_error(
    mock_connect, sample_df, mock_db_config, caplog
):
    """Tests handling of connection errors."""
    table_name = "conn_error_test"
    caplog.set_level(logging.ERROR)
    result = load.load_to_postgres(sample_df, mock_db_config, table_name)
    assert result is False
    mock_connect.assert_called_once_with(**mock_db_config)
    assert (
        f"PostgreSQL error during load to table '{table_name}': Connection failed"
        in caplog.text
    )


@patch("utils.load.psycopg2.connect")
@patch("utils.load._get_postgres_schema")
# No need to patch as_string as the error happens before insert query generation
def test_load_to_postgres_execute_error(
    mock_get_schema, mock_connect, sample_df, mock_db_config, caplog
):
    """Tests handling of errors during cursor execution (e.g., TRUNCATE)."""
    mock_conn, mock_cursor = setup_mock_conn_cursor(mock_connect)
    mock_schema_sql = sql.SQL('"col1" TEXT')
    mock_get_schema.return_value = mock_schema_sql
    table_name = "exec_error_test"
    caplog.set_level(logging.ERROR)
    truncate_error = Psycopg2Error("Truncate failed")

    def execute_side_effect(query, *args):
        query_str = str(query)
        if "TRUNCATE TABLE" in query_str:
            raise truncate_error
        return None

    mock_cursor.execute.side_effect = execute_side_effect
    result = load.load_to_postgres(sample_df, mock_db_config, table_name)
    assert result is False
    mock_get_schema.assert_called_once_with(sample_df)
    assert mock_cursor.execute.call_count >= 2
    mock_conn.commit.assert_not_called()
    mock_conn.rollback.assert_called_once()
    mock_conn.close.assert_called_once()
    assert (
        f"PostgreSQL error during load to table '{table_name}': Truncate failed"
        in caplog.text
    )


@patch("utils.load.psycopg2.connect")
@patch("utils.load._get_postgres_schema")
@patch("utils.load.execute_values", side_effect=Psycopg2Error("Insert failed"))
@patch("psycopg2.sql.Composed.as_string")
def test_load_to_postgres_execute_values_error(
    mock_as_string,
    mock_execute_values,
    mock_get_schema,
    mock_connect,
    sample_df,
    mock_db_config,
    caplog,
):
    """Tests handling of errors during bulk insert (execute_values)."""
    mock_conn, mock_cursor = setup_mock_conn_cursor(mock_connect)
    mock_schema_sql = sql.SQL('"col1" TEXT')
    mock_get_schema.return_value = mock_schema_sql
    table_name = "insert_error_test"
    caplog.set_level(logging.ERROR)
    # Configure mock as_string (needed to *reach* execute_values)
    mock_as_string.return_value = f'INSERT INTO "{table_name}" (...) VALUES %s'
    result = load.load_to_postgres(sample_df, mock_db_config, table_name)
    assert result is False
    mock_get_schema.assert_called_once_with(sample_df)
    assert mock_as_string.call_count > 0
    mock_execute_values.assert_called_once()
    mock_conn.commit.assert_not_called()
    mock_conn.rollback.assert_called_once()
    mock_conn.close.assert_called_once()
    assert (
        f"PostgreSQL error during load to table '{table_name}': Insert failed"
        in caplog.text
    )


@patch("utils.load.psycopg2.connect")
def test_load_to_postgres_missing_db_config_key(
    mock_connect, sample_df, mock_db_config, caplog
):
    """Tests handling of missing keys in the db_config dictionary."""
    incomplete_config = mock_db_config.copy()
    missing_key = "password"
    del incomplete_config[missing_key]
    table_name = "config_error_test"
    caplog.set_level(logging.ERROR)
    mock_connect.side_effect = KeyError(missing_key)
    result = load.load_to_postgres(sample_df, incomplete_config, table_name)
    assert result is False
    mock_connect.assert_called_once_with(**incomplete_config)
    assert f"Missing key in db_config: '{missing_key}'" in caplog.text


@patch("utils.load.psycopg2.connect")
@patch("utils.load._get_postgres_schema")
def test_load_to_postgres_unexpected_error_during_commit(
    mock_get_schema, mock_connect, sample_df, mock_db_config, caplog
):
    """Tests handling of unexpected non-DB errors during commit."""
    mock_conn, mock_cursor = setup_mock_conn_cursor(mock_connect)
    mock_schema_sql = "col1 TEXT"
    mock_get_schema.return_value = mock_schema_sql
    table_name = "unexpected_commit_err_test"
    caplog.set_level(logging.ERROR)

    # Mock SQL components
    mock_sql_obj = MagicMock()
    mock_sql_obj.as_string.return_value = f"INSERT INTO {table_name} (col1) VALUES %s"

    # Simulate commit failure
    mock_conn.commit.side_effect = Exception("Unexpected commit error")

    # Mock execute_values to ensure it doesn't fail
    with patch("utils.load.execute_values") as mock_execute_values, patch(
        "psycopg2.sql.SQL", return_value=mock_sql_obj
    ), patch("psycopg2.sql.Identifier", side_effect=lambda x: x):

        # Act
        result = load.load_to_postgres(sample_df, mock_db_config, table_name)

    # Assert
    assert result is False
    mock_connect.assert_called_once()
    mock_get_schema.assert_called_once()
    mock_conn.commit.assert_called_once()
    mock_conn.rollback.assert_called_once()
    mock_conn.close.assert_called_once()

    assert (
        f"An unexpected error occurred during PostgreSQL load to '{table_name}'"
        in caplog.text
    )
    assert "Unexpected commit error" in caplog.text
    assert "PostgreSQL transaction rolled back" in caplog.text


@patch("utils.load.psycopg2.connect")
@patch("utils.load._get_postgres_schema")
def test_load_to_postgres_unexpected_error_during_execute_values(
    mock_get_schema, mock_connect, sample_df, mock_db_config, caplog
):
    """Tests handling of unexpected non-DB errors during execute_values."""
    mock_conn, mock_cursor = setup_mock_conn_cursor(mock_connect)
    mock_schema_sql = "col1 TEXT"
    mock_get_schema.return_value = mock_schema_sql
    table_name = "unexpected_generic_err_test"
    caplog.set_level(logging.ERROR)

    # Mock SQL components
    mock_sql_obj = MagicMock()
    mock_sql_obj.as_string.return_value = f"INSERT INTO {table_name} (col1) VALUES %s"

    # Set up execute_values to raise the specific exception
    with patch(
        "utils.load.execute_values",
        side_effect=Exception("Unexpected generic error during execute_values"),
    ) as mock_execute_values, patch(
        "psycopg2.sql.SQL", return_value=mock_sql_obj
    ), patch(
        "psycopg2.sql.Identifier", side_effect=lambda x: x
    ):

        # Act
        result = load.load_to_postgres(sample_df, mock_db_config, table_name)

    # Assert
    assert result is False
    mock_connect.assert_called_once_with(**mock_db_config)
    mock_get_schema.assert_called_once_with(sample_df)
    mock_conn.rollback.assert_called_once()
    mock_conn.commit.assert_not_called()
    mock_conn.close.assert_called_once()

    assert (
        f"An unexpected error occurred during PostgreSQL load to '{table_name}'"
        in caplog.text
    )
    assert "Unexpected generic error during execute_values" in caplog.text
    assert "PostgreSQL transaction rolled back" in caplog.text
