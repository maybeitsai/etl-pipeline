# utils/extract.py
"""
Module for extracting product data from the Fashion Studio website.
Handles fetching HTML content and parsing product information, including pagination.
"""

import logging
import time
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException, Timeout

from utils.constants import LOG_FORMAT, REQUEST_DELAY, REQUEST_TIMEOUT, USER_AGENT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def _parse_product_details(
    details_container: BeautifulSoup, product_info: Dict[str, Optional[str]]
) -> None:
    """Parses rating, colors, size, and gender from the details container."""
    detail_paragraphs = details_container.find_all("p")
    found_details = set()
    for paragraph in detail_paragraphs:
        text = paragraph.text.strip()
        if "Rating:" in text and "rating" not in found_details:
            product_info["rating"] = text.split("Rating:", 1)[-1].strip()
            found_details.add("rating")
        elif "Colors" in text and "colors" not in found_details:
            product_info["colors"] = text  # Keep "N Colors" format
            found_details.add("colors")
        elif "Size:" in text and "size" not in found_details:
            product_info["size"] = text.split("Size:", 1)[-1].strip()
            found_details.add("size")
        elif "Gender:" in text and "gender" not in found_details:
            product_info["gender"] = text.split("Gender:", 1)[-1].strip()
            found_details.add("gender")


def _parse_product_card(
    card: BeautifulSoup, url: str
) -> Optional[Dict[str, Optional[str]]]:
    """
    Parses a single product card BeautifulSoup object into a dictionary.

    Args:
        card: The BeautifulSoup object representing the product card.
        url: The URL of the page where the card was found (for logging).

    Returns:
        A dictionary containing product information, or None if essential
        information (like title) is missing.
    """
    product_info: Dict[str, Optional[str]] = {
        "title": None,
        "price": None,
        "rating": None,
        "colors": None,
        "size": None,
        "gender": None,
        "image_url": None,
    }

    # Extract Product Name (Mandatory)
    title_tag = card.find("h3", class_="product-title")
    if not title_tag or not title_tag.text.strip():
        logging.warning(
            "Could not find valid product title for a card on %s. Skipping card.", url
        )
        return None
    product_info["title"] = title_tag.text.strip()

    # Extract Price
    price_tag = card.find("span", class_="price")
    if price_tag:
        product_info["price"] = price_tag.text.strip()
    else:
        # Check for explicit "Price Unavailable" text
        price_unavailable_tag = card.find("p", class_="price")
        if price_unavailable_tag and "Price Unavailable" in price_unavailable_tag.text:
            product_info["price"] = "Price Unavailable"
        else:
            logging.warning(
                "Could not find price for product '%s' on %s.",
                product_info["title"],
                url,
            )

    # Extract Image URL
    img_tag = card.find("img", class_="collection-image")
    if img_tag and img_tag.has_attr("src"):
        product_info["image_url"] = img_tag["src"]
    else:
        logging.warning(
            "Could not find image URL for product '%s' on %s.",
            product_info["title"],
            url,
        )

    # Extract Details (Rating, Colors, Size, Gender)
    details_container = card.find("div", class_="product-details")
    if details_container:
        _parse_product_details(details_container, product_info)
    else:
        logging.warning(
            "Could not find 'product-details' div for product '%s' on %s.",
            product_info["title"],
            url,
        )

    return product_info


def extract_product_data(url: str) -> Optional[List[Dict[str, Optional[str]]]]:
    """
    Extracts product data from a single page of the website.

    Args:
        url: The URL of the product listing page.

    Returns:
        A list of dictionaries, where each dictionary contains data for one product.
        Returns None if the request fails (e.g., timeout, HTTP error).
        Returns an empty list if the page structure is unexpected or no products found.
    """
    products = []
    try:
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()  # Raises HTTPError for bad responses (4XX, 5XX)
        logging.info("Successfully fetched HTML content from %s", url)
    except Timeout:
        logging.error("Timeout occurred while fetching URL %s", url)
        return None
    except RequestException as e:
        logging.error("Request failed for URL %s: %s", url, e)
        return None

    try:
        soup = BeautifulSoup(response.text, "html.parser")
        collection_list = soup.find("div", id="collectionList")

        if not collection_list:
            logging.warning(
                "Could not find collection list div with id='collectionList' on %s.",
                url,
            )
            return []  # Return empty list, as page was fetched but structure differs

        product_cards = collection_list.find_all("div", class_="collection-card")
        if not product_cards:
            logging.warning(
                "Found collection list, but no product cards inside on page %s.", url
            )
            return []

        logging.info("Found %d product cards on page %s.", len(product_cards), url)

        for card in product_cards:
            product_data = _parse_product_card(card, url)
            if product_data:
                products.append(product_data)

    # Catching a broad exception during parsing is often necessary due to
    # unpredictable HTML structures or potential BS4/parser issues.
    # Use exc_info=True for detailed traceback in logs.
    except Exception as e:  # pylint: disable=broad-except
        logging.error(
            "An error occurred during HTML parsing on page %s: %s",
            url,
            e,
            exc_info=True,
        )
        # Return data parsed so far, or empty list if parsing failed early
        return products

    logging.info(
        "Successfully extracted data for %d products from page %s.", len(products), url
    )
    return products


def scrape_all_pages(base_url: str, max_pages: int) -> List[Dict[str, Optional[str]]]:
    """
    Scrapes product data from multiple pages of the website.

    Args:
        base_url: The base URL (e.g., "https://fashion-studio.dicoding.dev").
        max_pages: The maximum number of pages to scrape.

    Returns:
        A list containing product data aggregated from all successfully scraped pages.
    """
    all_products: List[Dict[str, Optional[str]]] = []
    normalized_base_url = base_url.rstrip("/")

    for page_num in range(1, max_pages + 1):
        if page_num == 1:
            current_url = normalized_base_url + "/"
        else:
            current_url = f"{normalized_base_url}/page{page_num}"

        logging.info("--- Scraping Page %d: %s ---", page_num, current_url)
        page_data = extract_product_data(current_url)

        if page_data is None:
            logging.error(
                "Failed to fetch/process page %d (%s). Skipping.", page_num, current_url
            )
        elif not page_data:
            logging.warning(
                "No products found on page %d (%s). Possibly end of results.",
                page_num,
                current_url,
            )
            # Could add logic here to break early if desired
        else:
            all_products.extend(page_data)
            logging.info(
                "Accumulated %d products after scraping page %d.",
                len(all_products),
                page_num,
            )

        # Delay only if not the last page and if scraping continues
        if page_num < max_pages and page_data is not None:
            time.sleep(REQUEST_DELAY)

    logging.info(
        "Finished scraping %d pages. Total products extracted: %d",
        max_pages,
        len(all_products),
    )
    return all_products


# Example usage (optional, for testing module directly)
if __name__ == "__main__":
    # Define a test URL (replace if necessary)
    TEST_BASE_URL = "https://fashion-studio.dicoding.dev"
    MAX_TEST_PAGES = 2

    print(f"\n--- Testing Extraction on Base URL: {TEST_BASE_URL} ---")
    extracted_data_all = scrape_all_pages(TEST_BASE_URL, MAX_TEST_PAGES)

    if extracted_data_all:
        print(f"\nSuccessfully extracted {len(extracted_data_all)} products in total.")
        print("Sample of extracted data (first 2 products):")
        for i, prod in enumerate(extracted_data_all[:2]):
            print(f"Product {i+1}: {prod}")
    else:
        print("\nExtraction process completed, but no products were extracted.")

    print("\n--- Testing Single Page Extraction (Page 1) ---")
    extracted_page1 = extract_product_data(TEST_BASE_URL + "/")
    if extracted_page1 is not None:
        print(f"Extracted {len(extracted_page1)} products from page 1.")
    else:
        print("Extraction failed for page 1.")
