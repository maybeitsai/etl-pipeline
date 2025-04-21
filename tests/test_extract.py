# tests/test_extract.py
# (Tidak ada perubahan signifikan yang diperlukan untuk cakupan extract.py,
# karena baris yang hilang ada di blok __main__)
"""
Unit tests for the utils.extract module.
"""

import logging
import time
from unittest.mock import MagicMock, call, patch

import pytest
import requests
from bs4 import BeautifulSoup
from requests.exceptions import HTTPError, RequestException, Timeout

# Assuming utils is importable from the project root
from utils import extract
from utils.constants import REQUEST_DELAY, USER_AGENT

# Sample HTML structures for testing parsing logic
SAMPLE_CARD_HTML_FULL = """
<div class="collection-card">
    <img class="collection-image" src="image.jpg">
    <h3 class="product-title">  Awesome T-Shirt  </h3>
    <span class="price"> $19.99 </span>
    <div class="product-details">
        <p>Rating: ⭐ 4.5 / 5</p>
        <p>Available in 3 Colors</p>
        <p>Size: M </p>
        <p>Gender: Unisex</p>
    </div>
</div>
"""

SAMPLE_CARD_HTML_NO_PRICE = """
<div class="collection-card">
    <img class="collection-image" src="image2.jpg">
    <h3 class="product-title">Cool Hat</h3>
    <!-- No price span -->
    <div class="product-details">
        <p>Rating: ⭐ 4.0 / 5</p>
        <p>Gender: Men</p>
    </div>
</div>
"""

SAMPLE_CARD_HTML_PRICE_UNAVAILABLE = """
<div class="collection-card">
    <img class="collection-image" src="image3.jpg">
    <h3 class="product-title">Fancy Pants</h3>
    <p class="price">Price Unavailable</p>
    <div class="product-details">
        <p>Rating: ⭐ 3.5 / 5</p>
        <p>2 Colors</p>
        <p>Size: L</p>
    </div>
</div>
"""

SAMPLE_CARD_HTML_NO_IMAGE = """
<div class="collection-card">
    <!-- No img tag -->
    <h3 class="product-title">Invisible Cloak</h3>
    <span class="price">$99.99</span>
    <div class="product-details">
        <p>Rating: ⭐ 5.0 / 5</p>
    </div>
</div>
"""

SAMPLE_CARD_HTML_NO_DETAILS = """
<div class="collection-card">
    <img class="collection-image" src="image4.jpg">
    <h3 class="product-title">Basic Item</h3>
    <span class="price">$5.00</span>
    <!-- No product-details div -->
</div>
"""

SAMPLE_CARD_HTML_NO_TITLE = """
<div class="collection-card">
    <img class="collection-image" src="image5.jpg">
    <!-- No h3 product-title -->
    <span class="price">$15.00</span>
    <div class="product-details">
        <p>Rating: ⭐ 4.8 / 5</p>
    </div>
</div>
"""

SAMPLE_PAGE_HTML_MULTIPLE_CARDS = f"""
<html><body>
<div id="collectionList">
    {SAMPLE_CARD_HTML_FULL}
    {SAMPLE_CARD_HTML_NO_PRICE}
    {SAMPLE_CARD_HTML_PRICE_UNAVAILABLE}
</div>
</body></html>
"""

SAMPLE_PAGE_HTML_NO_LIST = """
<html><body>
<div>Some other content</div>
</body></html>
"""

SAMPLE_PAGE_HTML_EMPTY_LIST = """
<html><body>
<div id="collectionList">
    <!-- No collection-card divs -->
</div>
</body></html>
"""


# --- Tests for _parse_product_card ---


@pytest.mark.parametrize(
    "html, expected_dict, log_warnings",
    [
        (
            SAMPLE_CARD_HTML_FULL,
            {
                "title": "Awesome T-Shirt",
                "price": "$19.99",
                "rating": "⭐ 4.5 / 5",
                "colors": "Available in 3 Colors",
                "size": "M",
                "gender": "Unisex",
                "image_url": "image.jpg",
            },
            [],
        ),
        (
            SAMPLE_CARD_HTML_NO_PRICE,
            {
                "title": "Cool Hat",
                "price": None,
                "rating": "⭐ 4.0 / 5",
                "colors": None,
                "size": None,
                "gender": "Men",
                "image_url": "image2.jpg",
            },
            ["Could not find price for product 'Cool Hat'"],
        ),
        (
            SAMPLE_CARD_HTML_PRICE_UNAVAILABLE,
            {
                "title": "Fancy Pants",
                "price": "Price Unavailable",
                "rating": "⭐ 3.5 / 5",
                "colors": "2 Colors",
                "size": "L",
                "gender": None,
                "image_url": "image3.jpg",
            },
            [],
        ),
        (
            SAMPLE_CARD_HTML_NO_IMAGE,
            {
                "title": "Invisible Cloak",
                "price": "$99.99",
                "rating": "⭐ 5.0 / 5",
                "colors": None,
                "size": None,
                "gender": None,
                "image_url": None,
            },
            ["Could not find image URL for product 'Invisible Cloak'"],
        ),
        (
            SAMPLE_CARD_HTML_NO_DETAILS,
            {
                "title": "Basic Item",
                "price": "$5.00",
                "rating": None,
                "colors": None,
                "size": None,
                "gender": None,
                "image_url": "image4.jpg",
            },
            ["Could not find 'product-details' div for product 'Basic Item'"],
        ),
        (
            SAMPLE_CARD_HTML_NO_TITLE,
            None,  # Expect None when title is missing
            ["Could not find valid product title"],
        ),
    ],
)
def test_parse_product_card(html, expected_dict, log_warnings, caplog):
    """Tests _parse_product_card with various HTML structures."""
    # Arrange
    soup = BeautifulSoup(html, "html.parser")
    # Handle potential None card if HTML is invalid for find
    card = soup.find("div", class_="collection-card")
    if card is None and html == SAMPLE_CARD_HTML_NO_TITLE: # Special case for testing None return
         card = BeautifulSoup(html, "html.parser") # Pass the whole soup if card isn't found as expected

    test_url = "http://test.com/page1"
    caplog.set_level(logging.WARNING)

    # Act
    result = extract._parse_product_card(card, test_url)

    # Assert
    assert result == expected_dict
    # Check logs only if warnings are expected
    if log_warnings:
        for warning_msg in log_warnings:
            assert warning_msg in caplog.text


# --- Tests for extract_product_data ---


@patch("utils.extract.requests.get")
def test_extract_product_data_success(mock_get, caplog):
    """Tests successful extraction from a page with multiple products."""
    # Arrange
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = SAMPLE_PAGE_HTML_MULTIPLE_CARDS
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    test_url = "http://test.com/page1"
    caplog.set_level(logging.INFO)

    # Act
    result = extract.extract_product_data(test_url)

    # Assert
    mock_get.assert_called_once_with(
        test_url, headers={"User-Agent": USER_AGENT}, timeout=extract.REQUEST_TIMEOUT
    )
    mock_response.raise_for_status.assert_called_once()
    assert len(result) == 3
    assert result[0]["title"] == "Awesome T-Shirt"
    assert result[1]["title"] == "Cool Hat"
    assert result[2]["title"] == "Fancy Pants"
    assert f"Successfully fetched HTML content from {test_url}" in caplog.text
    assert f"Found 3 product cards on page {test_url}" in caplog.text
    assert f"Successfully extracted data for 3 products from page {test_url}" in caplog.text


@patch("utils.extract.requests.get")
def test_extract_product_data_request_timeout(mock_get, caplog):
    """Tests handling of requests.Timeout."""
    # Arrange
    test_url = "http://timeout.com"
    mock_get.side_effect = Timeout("Request timed out")
    caplog.set_level(logging.ERROR)

    # Act
    result = extract.extract_product_data(test_url)

    # Assert
    assert result is None
    assert f"Timeout occurred while fetching URL {test_url}" in caplog.text
    mock_get.assert_called_once_with(
        test_url, headers={"User-Agent": USER_AGENT}, timeout=extract.REQUEST_TIMEOUT
    )


@patch("utils.extract.requests.get")
def test_extract_product_data_http_error(mock_get, caplog):
    """Tests handling of HTTPError (e.g., 404 Not Found)."""
    # Arrange
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = HTTPError("404 Client Error")
    mock_get.return_value = mock_response
    test_url = "http://notfound.com"
    caplog.set_level(logging.ERROR)

    # Act
    result = extract.extract_product_data(test_url)

    # Assert
    assert result is None
    assert f"Request failed for URL {test_url}: 404 Client Error" in caplog.text
    mock_response.raise_for_status.assert_called_once()


@patch("utils.extract.requests.get")
def test_extract_product_data_other_request_exception(mock_get, caplog):
    """Tests handling of other RequestException errors."""
    # Arrange
    test_url = "http://connectionerror.com"
    mock_get.side_effect = RequestException("Connection error")
    caplog.set_level(logging.ERROR)

    # Act
    result = extract.extract_product_data(test_url)

    # Assert
    assert result is None
    assert f"Request failed for URL {test_url}: Connection error" in caplog.text


@patch("utils.extract.requests.get")
def test_extract_product_data_no_collection_list(mock_get, caplog):
    """Tests handling when the main collection list div is missing."""
    # Arrange
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = SAMPLE_PAGE_HTML_NO_LIST
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    test_url = "http://test.com/no-list"
    caplog.set_level(logging.WARNING)

    # Act
    result = extract.extract_product_data(test_url)

    # Assert
    assert result == []
    assert (
        f"Could not find collection list div with id='collectionList' on {test_url}."
        in caplog.text
    )


@patch("utils.extract.requests.get")
def test_extract_product_data_empty_collection_list(mock_get, caplog):
    """Tests handling when the collection list is present but empty."""
    # Arrange
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = SAMPLE_PAGE_HTML_EMPTY_LIST
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    test_url = "http://test.com/empty-list"
    caplog.set_level(logging.WARNING)

    # Act
    result = extract.extract_product_data(test_url)

    # Assert
    assert result == []
    assert (
        f"Found collection list, but no product cards inside on page {test_url}."
        in caplog.text
    )


@patch("utils.extract.requests.get")
@patch("utils.extract.BeautifulSoup")
def test_extract_product_data_parsing_error(mock_bs, mock_get, caplog):
    """Tests handling of unexpected errors during BeautifulSoup parsing."""
    # Arrange
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html"  # Incomplete HTML to potentially cause issues
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    mock_bs.side_effect = Exception("Parsing failed badly")
    test_url = "http://test.com/parse-error"
    caplog.set_level(logging.ERROR)

    # Act
    result = extract.extract_product_data(test_url)

    # Assert
    # Exception occurs during BeautifulSoup instantiation, before product list is populated
    assert result == [] # Function returns products list which is empty at this point
    assert f"An error occurred during HTML parsing on page {test_url}" in caplog.text
    assert "Parsing failed badly" in caplog.text


# --- Tests for scrape_all_pages ---


@patch("utils.extract.extract_product_data")
@patch("utils.extract.time.sleep")
def test_scrape_all_pages_success(mock_sleep, mock_extract, caplog):
    """Tests scraping multiple pages successfully."""
    # Arrange
    base_url = "http://example.com"
    max_pages = 3
    # Simulate extract_product_data returning different data for each page
    mock_extract.side_effect = [
        [{"title": "Page1 Prod1"}],
        [{"title": "Page2 Prod1"}, {"title": "Page2 Prod2"}],
        [{"title": "Page3 Prod1"}],
    ]
    caplog.set_level(logging.INFO)

    # Act
    result = extract.scrape_all_pages(base_url, max_pages)

    # Assert
    assert len(result) == 4
    assert result[0]["title"] == "Page1 Prod1"
    assert result[1]["title"] == "Page2 Prod1"
    assert result[3]["title"] == "Page3 Prod1"

    # Check calls to extract_product_data
    expected_calls = [
        call("http://example.com/"),
        call("http://example.com/page2"),
        call("http://example.com/page3"),
    ]
    mock_extract.assert_has_calls(expected_calls)
    assert mock_extract.call_count == max_pages

    # Check calls to time.sleep (should be max_pages - 1)
    assert mock_sleep.call_count == max_pages - 1
    mock_sleep.assert_called_with(REQUEST_DELAY)

    assert "Scraping Page 1: http://example.com/" in caplog.text
    assert "Scraping Page 2: http://example.com/page2" in caplog.text
    assert "Scraping Page 3: http://example.com/page3" in caplog.text
    assert "Accumulated 1 products after scraping page 1." in caplog.text
    assert "Accumulated 3 products after scraping page 2." in caplog.text
    assert "Accumulated 4 products after scraping page 3." in caplog.text
    assert f"Finished scraping {max_pages} pages." in caplog.text


@patch("utils.extract.extract_product_data")
@patch("utils.extract.time.sleep")
def test_scrape_all_pages_with_failures_and_empty(mock_sleep, mock_extract, caplog):
    """Tests scraping with a mix of successful, failed, and empty pages."""
    # Arrange
    base_url = "http://complex.com/"  # Test trailing slash handling
    max_pages = 4
    mock_extract.side_effect = [
        [{"title": "Page1 Prod1"}],  # Page 1 success
        None,  # Page 2 fails
        [],  # Page 3 empty
        [{"title": "Page4 Prod1"}],  # Page 4 success
    ]
    caplog.set_level(logging.INFO)

    # Act
    result = extract.scrape_all_pages(base_url, max_pages)

    # Assert
    assert len(result) == 2
    assert result[0]["title"] == "Page1 Prod1"
    assert result[1]["title"] == "Page4 Prod1"

    # Check calls to extract_product_data
    expected_calls = [
        call("http://complex.com/"),
        call("http://complex.com/page2"),
        call("http://complex.com/page3"),
        call("http://complex.com/page4"),
    ]
    mock_extract.assert_has_calls(expected_calls)
    assert mock_extract.call_count == max_pages

    # Check calls to time.sleep (called after page 1 success, not after page 2 fail, after page 3 empty/success)
    # Sleep is called if page_num < max_pages AND page_data is not None.
    # Page 1: page_data is not None -> sleep
    # Page 2: page_data is None -> no sleep
    # Page 3: page_data is not None (it's []) -> sleep
    assert mock_sleep.call_count == 2
    mock_sleep.assert_has_calls([call(REQUEST_DELAY), call(REQUEST_DELAY)])

    assert "Failed to fetch/process page 2 (http://complex.com/page2)" in caplog.text
    assert "No products found on page 3 (http://complex.com/page3)" in caplog.text
    assert "Accumulated 1 products after scraping page 1." in caplog.text
    # No accumulation message for page 2 or 3
    assert "Accumulated 2 products after scraping page 4." in caplog.text
    assert f"Finished scraping {max_pages} pages." in caplog.text


@patch("utils.extract.extract_product_data")
@patch("utils.extract.time.sleep")
def test_scrape_all_pages_single_page(mock_sleep, mock_extract):
    """Tests scraping only one page."""
    # Arrange
    base_url = "http://single.com"
    max_pages = 1
    mock_extract.return_value = [{"title": "SinglePageProd"}]

    # Act
    result = extract.scrape_all_pages(base_url, max_pages)

    # Assert
    assert len(result) == 1
    mock_extract.assert_called_once_with("http://single.com/")
    mock_sleep.assert_not_called()  # No delay after the last page