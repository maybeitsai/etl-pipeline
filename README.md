# ETL Pipeline for Fashion Product Data

## Overview

This project implements an Extract, Transform, Load (ETL) pipeline designed to scrape product information from the [Fashion Studio Dicoding](https://fashion-studio.dicoding.dev) website. It extracts raw product data, cleans and transforms it into a structured format, and loads the final dataset into multiple destinations: CSV files, Google Sheets, and a PostgreSQL database.

The pipeline is built using Python 3.9 and leverages libraries like `requests`, `beautifulsoup4`, `pandas`, `psycopg2`, and `gspread`. Configuration is managed through environment variables.

## Features

*   **Web Scraping:** Extracts product details (title, price, rating, colors, size, gender, image URL) from multiple pages using pagination.
*   **Data Transformation:**
    *   Cleans raw text data (stripping whitespace, parsing numbers).
    *   Handles missing or invalid data points (e.g., "Price Unavailable", "Not Rated").
    *   Converts prices from USD to IDR based on a defined rate.
    *   Adds a timestamp (`Asia/Jakarta` timezone) for tracking when the data was processed.
    *   Enforces a final data schema and data types.
    *   Removes duplicate entries and rows with missing essential information.
*   **Data Loading:** Supports loading the transformed data into:
    *   CSV files.
    *   Google Sheets worksheets.
    *   PostgreSQL database tables.
*   **Configuration:** Uses environment variables (via a `.env` file) for flexible setup (URLs, credentials, file paths, database details).
*   **Logging:** Provides informative logging throughout the ETL process.
*   **Testing:** Includes unit tests with coverage reporting using `pytest`.

## Project Structure

```
etl-pipeline/
├── .env                   # Example environment variables file
├── main.py                # Main script to run the ETL pipeline
├── requirements.txt       # Project dependencies
├── utils/                 # Core ETL logic modules
│   ├── __init__.py
│   ├── constants.py       # Pipeline constants (URLs, schema, rates)
│   ├── extract.py         # Data extraction logic (web scraping)
│   ├── load.py            # Data loading logic (CSV, GSheets, Postgres)
│   └── transform.py       # Data transformation logic
└── tests/                 # Unit tests for utils modules
    ├── __init__.py
    ├── test_extract.py
    ├── test_load.py
    └── test_transform.py
```

## Prerequisites

*   Python 3.9 or higher
*   Pip (Python package installer)
*   Git
*   **Optional (for specific load targets):**
    *   PostgreSQL Server (if loading to PostgreSQL)
    *   Google Cloud Platform Account with Google Sheets API enabled and Service Account credentials (if loading to Google Sheets)

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/maybeitsai/etl-pipeline.git
    cd etl-pipeline
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # On Windows
    python -m venv venv
    .\venv\Scripts\activate

    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The pipeline requires configuration via environment variables.

1.  **Create a `.env` file:**
    Copy the example file:
    ```bash
    cp .env.example .env
    ```

2.  **Edit the `.env` file:**
    Fill in the necessary values.

    ```dotenv
    # --- REQUIRED ---
    # Source website URL
    SOURCE_URL="https://fashion-studio.dicoding.dev"

    # --- EXTRACTION (Optional - Defaults Provided) ---
    # Maximum number of pages to scrape
    MAX_PAGES_TO_SCRAPE="50"

    # --- LOAD TARGETS (Configure as needed) ---

    # CSV Output File Path (Default: products.csv)
    CSV_FILEPATH="data/output_products.csv"

    # Google Sheets Configuration (Optional)
    # Path to your Google Service Account JSON key file
    GOOGLE_SHEETS_CREDENTIALS_PATH="/path/to/your/credentials.json"
    # The ID of the target Google Sheet (from its URL)
    GOOGLE_SHEET_ID="YOUR_GOOGLE_SHEET_ID"
    # Name of the worksheet to use/create (Default: products)
    WORKSHEET_NAME="FashionData"

    # PostgreSQL Configuration (Optional)
    DB_HOST="localhost"
    DB_PORT="5432"
    DB_NAME="your_database_name"
    DB_USER="your_database_user"
    DB_PASSWORD="your_database_password"
    # Name of the table to use/create (Default: products)
    POSTGRES_TABLE_NAME="fashion_products"
    ```

    *   `SOURCE_URL`: **Required.** The base URL of the website to scrape.
    *   `MAX_PAGES_TO_SCRAPE`: Optional. Maximum number of pages to scrape. Defaults to `50`.
    *   `CSV_FILEPATH`: Optional. Path where the output CSV file will be saved. Defaults to `products.csv` in the project root.
    *   `GOOGLE_SHEETS_CREDENTIALS_PATH`: Required for Google Sheets loading. Path to the service account JSON key file.
    *   `GOOGLE_SHEET_ID`: Required for Google Sheets loading. The ID of the target spreadsheet.
    *   `WORKSHEET_NAME`: Optional. Name of the worksheet within the Google Sheet. Defaults to `products`.
    *   `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`: Required for PostgreSQL loading. Connection details for your PostgreSQL database.
    *   `POSTGRES_TABLE_NAME`: Optional. Name of the table in the PostgreSQL database. Defaults to `products`.

    **Note:** If credentials or required IDs for Google Sheets or PostgreSQL are not provided, the respective load steps will be skipped.

## Usage

To run the full ETL pipeline, execute the main script from the project's root directory:

```bash
python main.py
```

The script will:
1.  Load configuration from the `.env` file.
2.  Run the extraction process based on `SOURCE_URL` and `MAX_PAGES_TO_SCRAPE`.
3.  Run the transformation process on the extracted data.
4.  Attempt to load the transformed data into the configured destinations (CSV, Google Sheets, PostgreSQL).

Logs will be printed to the console indicating the progress and results of each step.

## Testing

Unit tests are provided in the `tests/` directory. They use `pytest` and `pytest-cov` for coverage analysis.

1.  **Run all tests:**
    ```bash
    pytest tests
    ```

2.  **Run tests with coverage report:**
    ```bash
    pytest tests --cov=utils
    ```

3.  **View detailed coverage report:**
    ```bash
    coverage report -m
    ```

## Modules

### `main.py`

*   **Purpose:** Entry point for the ETL pipeline.
*   **Functionality:**
    *   Loads configuration from environment variables using `python-dotenv`.
    *   Orchestrates the execution of the Extract, Transform, and Load steps by calling functions from the `utils` modules.
    *   Handles overall pipeline logging and error management.

### `utils/constants.py`

*   **Purpose:** Centralizes constant values used across the pipeline.
*   **Key Constants:**
    *   `USER_AGENT`, `REQUEST_TIMEOUT`, `REQUEST_DELAY`: For web scraping politeness and control.
    *   `USD_TO_IDR_RATE`: Currency conversion rate.
    *   `FINAL_SCHEMA_TYPE_MAPPING`, `REQUIRED_COLUMNS`: Defines the expected structure and non-nullable fields of the transformed data.
    *   `LOG_FORMAT`: Standard logging format.
    *   `CSV_FILEPATH`, `WORKSHEET_NAME`, `POSTGRES_TABLE_NAME`: Default names/paths for load targets.

### `utils/extract.py`

*   **Purpose:** Handles the data extraction (web scraping) process.
*   **Functionality:**
    *   `scrape_all_pages(base_url, max_pages)`: Iterates through website pages up to `max_pages`, calling `extract_product_data` for each.
    *   `extract_product_data(url)`: Fetches HTML content from a single URL, parses it using `BeautifulSoup`, finds product cards, and extracts raw data using `_parse_product_card`.
    *   `_parse_product_card(card, url)`: Parses a single product card HTML element to extract fields like title, price, rating, etc. Handles cases where elements might be missing.
    *   Uses `requests` for HTTP calls and includes error handling for network issues.

### `utils/transform.py`

*   **Purpose:** Cleans, validates, and restructures the raw extracted data.
*   **Functionality:**
    *   `transform_data(extracted_data)`: Main transformation function orchestrating various steps. Converts the input list of dictionaries into a Pandas DataFrame and applies cleaning, validation, and schema enforcement.
    *   `clean_price`, `clean_rating`, `clean_colors`: Helper functions to parse and clean specific fields using regular expressions and type conversions.
    *   `_initial_clean_and_parse`: Applies basic string stripping and initial cleaning functions.
    *   `_add_timestamp`: Adds a processing timestamp (timezone-aware).
    *   `_filter_invalid_rows`: Removes rows based on specific criteria (e.g., "Unknown Product" title).
    *   `_apply_business_logic`: Performs calculations like currency conversion.
    *   `_prepare_final_schema`: Selects final columns, renames them, and enforces the target data types defined in `constants.py`.
    *   `_remove_nulls_and_duplicates`: Drops rows with nulls in required columns and removes exact duplicate rows.

### `utils/load.py`

*   **Purpose:** Loads the transformed Pandas DataFrame into various destinations.
*   **Functionality:**
    *   `load_to_csv(df, filepath)`: Saves the DataFrame to a CSV file. Overwrites existing files. Handles directory creation.
    *   `load_to_gsheets(df, credentials_path, sheet_id, worksheet_name)`: Loads the DataFrame into a specified Google Sheet worksheet. Clears the existing worksheet content before loading. Handles authentication using service account credentials and worksheet creation if it doesn't exist.
    *   `load_to_postgres(df, db_config, table_name)`: Loads the DataFrame into a PostgreSQL table. Creates the table based on DataFrame schema if it doesn't exist, truncates existing data, and performs a bulk insert. Handles database connection and transactions.
    *   Includes robust error handling for file I/O, API interactions, and database operations.

## Contributing

Contributions are welcome! Please follow these general guidelines:

1.  **Fork the repository.**
2.  **Create a new branch** for your feature or bug fix: `git checkout -b feature/your-feature-name` or `git checkout -b fix/your-bug-fix`.
3.  **Make your changes.** Ensure code follows existing style and includes tests where appropriate.
4.  **Run tests** to ensure nothing is broken: `pytest tests --cov=utils`.
5.  **Commit your changes** with clear and descriptive messages.
6.  **Push your branch** to your fork: `git push origin feature/your-feature-name`.
7.  **Create a Pull Request** against the `main` branch of the original repository.

Please open an issue first to discuss significant changes or new features.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
