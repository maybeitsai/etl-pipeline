# utils/extract.py
import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Optional
import time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def extract_product_data(url: str) -> Optional[List[Dict[str, Optional[str]]]]:
    """
    Extracts product data from a single page of the Fashion Studio website.

    Args:
        url (str): The URL of the product listing page.

    Returns:
        Optional[List[Dict[str, Optional[str]]]]: A list of dictionaries,
        where each dictionary contains data for one product.
        Returns None if extraction fails at the request level.
        Returns an empty list if the page structure is unexpected.
    """
    products = []
    try:
        # Tambahkan header User-Agent agar terlihat seperti browser biasa
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        logging.info(f"Successfully fetched HTML content from {url}")
    except requests.exceptions.Timeout:
        logging.error(f"Timeout occurred while fetching URL {url}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching URL {url}: {e}")
        return None

    try:
        soup = BeautifulSoup(response.text, "html.parser")
        collection_list = soup.find("div", id="collectionList")

        # Jika tidak ada list produk di halaman ini (mungkin halaman terakhir atau error), kembalikan list kosong
        if not collection_list:
            logging.warning(
                f"Could not find the collection list div with id='collectionList' on page {url}."
            )
            return []

        # Kembalikan list kosong jika tidak ada kartu produk
        product_cards = collection_list.find_all("div", class_="collection-card")
        if not product_cards:
            logging.warning(
                f"Found collection list div, but no product cards inside on page {url}."
            )
            return []

        logging.info(f"Found {len(product_cards)} product cards on page {url}.")

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
            else:
                logging.warning(
                    "Could not find product title for a card. Skipping card."
                )
                continue

            # Extract Price
            price_tag = card.find("span", class_="price")
            if price_tag:
                product_info["price"] = price_tag.text.strip()
            else:
                # Handle cases like "Price Unavailable" which is in a <p> tag
                price_unavailable_tag = card.find("p", class_="price")
                if (
                    price_unavailable_tag
                    and "Price Unavailable" in price_unavailable_tag.text
                ):
                    product_info["price"] = "Price Unavailable"
                else:
                    # Jika tidak ada span atau p, log sebagai warning
                    logging.warning(
                        f"Could not find price for product: {product_info['product_name']}"
                    )
                    product_info["price"] = None

            # Extract Image URL
            img_tag = card.find("img", class_="collection-image")
            if img_tag and img_tag.has_attr("src"):
                product_info["image_url"] = img_tag["src"]
            else:
                logging.warning(
                    f"Could not find image URL for product: {product_info['product_name']}"
                )

            # --- PERBAIKAN EKSTRAKSI DETAIL (RATING, COLORS, SIZE, GENDER) ---
            details_container = card.find("div", class_="product-details")
            if details_container:
                # Temukan SEMUA tag <p> yang relevan (berdasarkan style atau cukup semua <p> di dalam details)
                # Menggunakan style lebih spesifik tapi bisa rapuh jika style berubah sedikit
                # detail_paragraphs = details_container.find_all('p', style="font-size: 14px; color: #777")
                # Alternatif: Ambil semua <p> di dalam product-details
                detail_paragraphs = details_container.find_all("p")

                for p in detail_paragraphs:
                    text = p.text.strip()
                    # Gunakan 'in' untuk pencarian yang lebih fleksibel
                    if "Rating:" in text:
                        # Ambil bagian setelah "Rating:", lalu strip spasi
                        rating_value = text.split("Rating:", 1)[-1].strip()
                        product_info["rating"] = rating_value
                    elif "Colors" in text:  # Cek kata "Colors"
                        # Ambil seluruh teksnya karena formatnya "N Colors"
                        product_info["colors"] = text
                    elif "Size:" in text:
                        size_value = text.split("Size:", 1)[-1].strip()
                        product_info["size"] = size_value
                    elif "Gender:" in text:
                        gender_value = text.split("Gender:", 1)[-1].strip()
                        product_info["gender"] = gender_value
                    # else: (Abaikan <p> lain seperti <p class="price"> jika menggunakan find_all('p'))
                    #    logging.debug(f"Ignoring paragraph: {text}")

            else:
                logging.warning(
                    f"Could not find 'product-details' div for product: {product_info['product_name']}"
                )

            # Hanya tambahkan jika produk punya nama (validasi dasar)
            products.append(product_info)
            # logging.debug(f"Extracted: {product_info}") # Debug untuk melihat hasil per produk

    except Exception as e:
        logging.error(
            f"An error occurred during HTML parsing on page {url}: {e}", exc_info=True
        )  # Tambahkan exc_info=True
        # Kembalikan apa yang sudah terkumpul sejauh ini atau list kosong
        return products if products else []

    logging.info(
        f"Successfully extracted data for {len(products)} products from page {url}."
    )
    return products


# --- Fungsi baru untuk mengelola pagination ---
def scrape_all_pages(
    base_url: str, max_pages: int = 50
) -> List[Dict[str, Optional[str]]]:
    """
    Scrapes product data from all pages up to max_pages.

    Args:
        base_url (str): The base URL (without trailing slash usually).
        max_pages (int): The maximum number of pages to scrape.

    Returns:
        List[Dict[str, Optional[str]]]: A list containing product data from all pages.
    """
    all_products = []
    if not base_url.endswith("/"):  # Pastikan base_url diakhiri slash
        base_url += "/"

    for page_num in range(1, max_pages + 1):
        if page_num == 1:
            current_url = base_url
        else:
            current_url = f"{base_url}page{page_num}"

        logging.info(f"--- Scraping Page {page_num}: {current_url} ---")
        page_data = extract_product_data(current_url)

        if page_data is None:
            logging.error(
                f"Failed to fetch or process page {page_num}. Stopping pagination."
            )
            # Anda bisa memilih untuk berhenti atau lanjut ke halaman berikutnya
            # break # Berhenti jika satu halaman gagal total
            continue  # Coba lanjut ke halaman berikutnya

        if not page_data:
            logging.warning(
                f"No products found on page {page_num}. Possibly end of results or page structure issue."
            )
            # Bisa jadi ini akhir dari produk, coba beberapa halaman lagi atau berhenti
            # Misalnya, jika 3 halaman berturut-turut kosong, mungkin berhenti.
            # Untuk sekarang, kita lanjutkan saja.

        all_products.extend(page_data)
        logging.info(f"Accumulated {len(all_products)} products so far.")

        # Tambahkan jeda sedikit antar request agar tidak membebani server
        time.sleep(0.5)  # Jeda 0.5 detik

    logging.info(
        f"Finished scraping all pages. Total products extracted: {len(all_products)}"
    )
    return all_products


# Example usage (optional, for testing module directly)
# if __name__ == '__main__':
#     target_base_url = "https://fashion-studio.dicoding.dev"
#     # Coba scrape beberapa halaman saja untuk testing
#     # products_from_all_pages = scrape_all_pages(target_base_url, max_pages=3)

#     # Atau scrape satu halaman spesifik untuk debug ekstraksi detail
#     test_url_page_1 = "https://fashion-studio.dicoding.dev/"
#     print(f"\n--- Testing Extraction on Page 1: {test_url_page_1} ---")
#     extracted_data_p1 = extract_product_data(test_url_page_1)
#     if extracted_data_p1:
#         print(f"Extracted {len(extracted_data_p1)} products from page 1.")
#         # Tampilkan beberapa produk untuk diperiksa detailnya
#         for i, prod in enumerate(extracted_data_p1[:5]): # Tampilkan 5 pertama
#              print(f"Product {i+1}: {prod}")
#     else:
#         print("Extraction failed for page 1.")

#     test_url_page_2 = "https://fashion-studio.dicoding.dev/page2"
#     print(f"\n--- Testing Extraction on Page 2: {test_url_page_2} ---")
#     extracted_data_p2 = extract_product_data(test_url_page_2)
#     if extracted_data_p2:
#         print(f"Extracted {len(extracted_data_p2)} products from page 2.")
#         for i, prod in enumerate(extracted_data_p2[:5]): # Tampilkan 5 pertama
#              print(f"Product {i+1}: {prod}")
#     else:
#         print("Extraction failed for page 2.")
