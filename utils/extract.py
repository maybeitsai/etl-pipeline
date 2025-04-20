import requests
from bs4 import BeautifulSoup
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_products(url: str) -> list[dict]:
    """
    Mengambil data produk dari URL target.

    Args:
        url (str): URL website target.

    Returns:
        list[dict]: List dictionary berisi data produk mentah.
                    Mengembalikan list kosong jika terjadi error.
    """
    products_data = []
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Raise HTTPError untuk status code 4xx/5xx
        logging.info(f"Berhasil mengambil data dari {url}")

        soup = BeautifulSoup(response.text, 'lxml')

        # --- INSPEKSI WEBSITE TARGET UNTUK MENENTUKAN SELEKTOR YANG TEPAT ---
        # Contoh selektor (INI HARUS DISESUAIKAN berdasarkan struktur HTML aktual):
        product_cards = soup.find_all('div', class_='product-card') # Ganti dengan selektor yang benar

        if not product_cards:
            logging.warning("Tidak ada kartu produk yang ditemukan dengan selektor yang diberikan.")
            return []

        for card in product_cards:
            try:
                # Contoh ekstraksi (GANTI DENGAN SELEKTOR YANG BENAR)
                name = card.find('h3', class_='product-name').text.strip()
                price_text = card.find('span', class_='product-price').text.strip()
                # Kategori mungkin perlu logika tambahan (misal dari URL atau elemen lain)
                category = "Unknown" # Placeholder
                product_url = card.find('a')['href'] # Contoh

                products_data.append({
                    'name': name,
                    'price_raw': price_text, # Simpan mentah dulu, transform nanti
                    'category': category,
                    'url': product_url # Mungkin perlu di-resolve ke URL absolut
                })
            except AttributeError as e:
                logging.warning(f"Gagal mengekstrak data dari satu kartu produk: {e} - Card: {card.prettify()[:200]}...")
                continue # Lanjut ke kartu berikutnya

    except requests.exceptions.Timeout:
        logging.error(f"Request timeout saat mencoba mengakses {url}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error saat request ke {url}: {e}")
    except Exception as e:
        logging.error(f"Terjadi error tidak terduga saat ekstraksi: {e}")

    logging.info(f"Ekstraksi selesai. Ditemukan {len(products_data)} produk.")
    return products_data