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

from utils.constants import LOG_FORMAT, REQUEST_DELAY, REQUEST_TIMEOUT, USER_AGENT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def _parse_product_card(
    card: BeautifulSoup, url: str
) -> Optional[Dict[str, Optional[str]]]:
    """Parses a single product card BeautifulSoup object."""
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
    if not title_tag:
        logging.warning(
            "Could not find product title for a card on %s. Skipping card.", url
        )
        return None
    product_info["title"] = title_tag.text.strip()

    # Extract Price
    price_tag = card.find("span", class_="price")
    if price_tag:
        product_info["price"] = price_tag.text.strip()
    else:
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
        detail_paragraphs = details_container.find_all("p")
        for paragraph in detail_paragraphs:
            text = paragraph.text.strip()
            if "Rating:" in text:
                product_info["rating"] = text.split("Rating:", 1)[-1].strip()
            elif "Colors" in text:
                product_info["colors"] = text  # Keep "N Colors" format
            elif "Size:" in text:
                product_info["size"] = text.split("Size:", 1)[-1].strip()
            elif "Gender:" in text:
                product_info["gender"] = text.split("Gender:", 1)[-1].strip()
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
        Returns an empty list if the page structure is unexpected or no products are found.
    """
    products = []
    try:
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()  # Raises HTTPError for bad responses (4XX, 5XX)
        logging.info("Successfully fetched HTML content from %s", url)
    except requests.exceptions.Timeout:
        logging.error("Timeout occurred while fetching URL %s", url)
        return None
    except requests.exceptions.RequestException as e:
        logging.error("Error fetching URL %s: %s", url, e)
        return None

    try:
        soup = BeautifulSoup(response.text, "html.parser")
        collection_list = soup.find("div", id="collectionList")

        if not collection_list:
            logging.warning(
                "Could not find collection list div with id='collectionList' on page %s.",
                url,
            )
            return []  # Return empty list, not None, as the page was fetched

        product_cards = collection_list.find_all("div", class_="collection-card")
        if not product_cards:
            logging.warning(
                "Found collection list div, but no product cards inside on page %s.",
                url,
            )
            return []

        logging.info("Found %d product cards on page %s.", len(product_cards), url)

        for card in product_cards:
            product_data = _parse_product_card(card, url)
            if product_data:
                products.append(product_data)

    except Exception as e:
        # Catch potential parsing errors
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
        base_url: The base URL of the website (e.g., "https://fashion-studio.dicoding.dev").
        max_pages: The maximum number of pages to attempt scraping.

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
                "Failed to fetch or process page %d (%s). Skipping page.",
                page_num,
                current_url,
            )
            # Continue to the next page attempt, maybe it's a temporary issue
            time.sleep(REQUEST_DELAY)  # Still wait before next attempt
            continue

        if not page_data:
            logging.warning(
                "No products found on page %d (%s). "
                "This might indicate the end of results or a page structure issue.",
                page_num,
                current_url,
            )
            # Decide whether to stop early if no products found for consecutive pages?
            # For now, continue up to max_pages as requested.

        all_products.extend(page_data)
        logging.info(
            "Accumulated %d products after scraping page %d.",
            len(all_products),
            page_num,
        )

        # Add delay between requests to avoid overwhelming the server
        if page_num < max_pages:
            time.sleep(REQUEST_DELAY)

    logging.info(
        "Finished scraping up to %d pages. Total products extracted: %d",
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
        print("Sample of extracted data (first 5 products):")
        for i, prod in enumerate(extracted_data_all[:5]):
            print(f"Product {i+1}: {prod}")
    else:
        print("\nExtraction process completed, but no products were extracted.")

    print("\n--- Testing Single Page Extraction (Page 1) ---")
    extracted_page1 = extract_product_data(TEST_BASE_URL + "/")
    if extracted_page1 is not None:
        print(f"Extracted {len(extracted_page1)} products from page 1.")
    else:
        print("Extraction failed for page 1.")
