# utils/extract.py
import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def extract_product_data(url: str) -> Optional[List[Dict[str, Optional[str]]]]:
    """
    Extracts product data from the Fashion Studio website.

    Args:
        url (str): The URL of the product listing page.

    Returns:
        Optional[List[Dict[str, Optional[str]]]]: A list of dictionaries,
        where each dictionary contains data for one product.
        Returns None if extraction fails at the request level.
    """
    products = []
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        logging.info(f"Successfully fetched HTML content from {url}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching URL {url}: {e}")
        return None

    try:
        soup = BeautifulSoup(response.text, "html.parser")
        collection_list = soup.find("div", id="collectionList")

        # Return empty list if container not found
        if not collection_list:
            logging.warning(
                "Could not find the collection list div with id='collectionList'."
            )
            return []

        product_cards = collection_list.find_all("div", class_="collection-card")
        logging.info(f"Found {len(product_cards)} product cards.")

        for card in product_cards:
            product_info: Dict[str, Optional[str]] = {
                "product_name": None,
                "price": None,
                "rating": None,
                "colors": None,
                "size": None,
                "gender": None,
                "image_url": None,
            }

            # Extract Product Name
            title_tag = card.find("h3", class_="product-title")
            if title_tag:
                product_info["product_name"] = title_tag.text.strip()

            # Extract Price
            price_tag = card.find("span", class_="price")
            if price_tag:
                product_info["price"] = price_tag.text.strip()
            else:
                # Handle cases like "Price Unavailable"
                price_unavailable_tag = card.find("p", class_="price")
                if (
                    price_unavailable_tag
                    and "Price Unavailable" in price_unavailable_tag.text
                ):
                    product_info["price"] = "Price Unavailable"

            # Extract Details (Rating, Colors, Size, Gender) using paragraphs
            details = card.find_all("p", style="font-size: 14px; color: #777")
            for p in details:
                text = p.text.strip()
                if text.startswith("Rating:"):
                    product_info["rating"] = text.replace("Rating:", "").strip()
                elif text.endswith("Colors"):
                    product_info["colors"] = text
                elif text.startswith("Size:"):
                    product_info["size"] = text.replace("Size:", "").strip()
                elif text.startswith("Gender:"):
                    product_info["gender"] = text.replace("Gender:", "").strip()

            # Handle cases like "Rating: Not Rated" separately if needed
            if not product_info["rating"]:
                rating_not_rated_tag = card.find(
                    "p",
                    string=lambda t: t and t.strip().startswith("Rating: Not Rated"),
                )
                if rating_not_rated_tag:
                    product_info["rating"] = "Not Rated"

            # Extract Image URL
            img_tag = card.find("img", class_="collection-image")
            if img_tag and img_tag.has_attr("src"):
                product_info["image_url"] = img_tag["src"]

            # Only add if product name is found (basic validation)
            if product_info["product_name"]:
                products.append(product_info)
            else:
                logging.warning(
                    f"Skipping card due to missing product name. Card content: {card.text[:100]}..."
                )

    except Exception as e:
        logging.error(f"An error occurred during HTML parsing: {e}")
        return products

    logging.info(f"Successfully extracted data for {len(products)} products.")
    return products


# # Example usage (optional, for testing module directly)
# if __name__ == '__main__':
#     target_url = "https://fashion-studio.dicoding.dev/"
#     extracted_data = extract_product_data(target_url)
#     if extracted_data:
#         print(f"Extracted {len(extracted_data)} products.")
#         # print(extracted_data[:2]) # Print first two products
#     else:
#         print("Extraction failed.")
