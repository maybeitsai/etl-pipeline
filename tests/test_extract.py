# tests/test_extract.py

import pytest
import requests
import requests_mock
from bs4 import BeautifulSoup
import logging

# Pastikan path ke utils benar, sesuaikan jika perlu
from utils.extract import extract_products

# Contoh HTML (sesuaikan dengan struktur HTML target SEBENARNYA setelah inspeksi)
# Ini adalah contoh SANGAT sederhana, Anda HARUS menyesuaikannya
MOCK_HTML_SUCCESS = """
<html><body>
  <div class="product-card">
    <a href="/product/t-shirt-keren">
      <h3 class="product-name"> Kaos Polos Keren  </h3>
    </a>
    <span class="product-price"> Rp 75.000 </span>
    <!-- Anggap kategori ada di elemen lain atau URL -->
  </div>
  <div class="product-card">
    <a href="/product/jaket-bomber">
      <h3 class="product-name">Jaket Bomber Stylish </h3>
    </a>
    <span class="product-price"> Rp 250.000 </span>
  </div>
  <div class="product-card">
    <!-- Produk tanpa harga -->
    <a href="/product/celana-error">
      <h3 class="product-name">Celana Error</h3>
    </a>
  </div>
</body></html>
"""

MOCK_HTML_NO_PRODUCTS = """
<html><body>
  <h1>Tidak ada produk ditemukan</h1>
</body></html>
"""

TARGET_URL = "https://fashion-studio.dicoding.dev" # Atau URL yang sama di extract.py

def test_extract_products_success(requests_mock):
    """Test ekstraksi berhasil dengan data yang valid."""
    requests_mock.get(TARGET_URL, text=MOCK_HTML_SUCCESS, status_code=200)

    extracted_data = extract_products(TARGET_URL)

    assert len(extracted_data) == 3 # 3 kartu produk di HTML mock

    # Periksa data produk pertama (sesuaikan dengan ekstraksi Anda)
    assert extracted_data[0]['name'] == 'Kaos Polos Keren'
    assert extracted_data[0]['price_raw'] == 'Rp 75.000'
    # Asumsikan URL relatif dan kategori default (sesuaikan jika logika berbeda)
    assert extracted_data[0]['url'] == '/product/t-shirt-keren'
    assert extracted_data[0]['category'] == 'Unknown' # Sesuai implementasi extract.py

    # Periksa produk kedua
    assert extracted_data[1]['name'] == 'Jaket Bomber Stylish'
    assert extracted_data[1]['price_raw'] == 'Rp 250.000'

    # Periksa produk ketiga (yang tidak punya harga)
    assert extracted_data[2]['name'] == 'Celana Error'
    # Pastikan handling error di extract.py (misal, price jadi None atau string kosong)
    # Jika extract.py men-skip produk error, assert len(extracted_data) == 2
    # Dalam contoh extract.py kita, dia akan mencoba find price dan gagal, jadi data lain tetap ada
    # Tapi price_raw mungkin tidak ada jika find gagal. Mari kita asumsikan ia skip.
    # Modifikasi extract.py agar lebih robust jika find gagal.
    # Untuk contoh ini, anggap dia skip jika find('span', class_='product-price') gagal
    # ---> PERLU MODIFIKASI extract.py agar test ini pass <---
    # Misal, di extract.py, dalam loop for card:, tambahkan:
    # price_element = card.find('span', class_='product-price')
    # if not price_element:
    #     logging.warning(f"Harga tidak ditemukan untuk produk: {name}. Skipping.")
    #     continue # Skip produk ini
    # price_text = price_element.text.strip()
    # --> Jika modifikasi di atas dilakukan, maka assert len(extracted_data) harus jadi 2 <--


def test_extract_products_no_products_found(requests_mock, caplog):
    """Test ketika tidak ada elemen produk yang cocok dengan selektor."""
    requests_mock.get(TARGET_URL, text=MOCK_HTML_NO_PRODUCTS, status_code=200)

    with caplog.at_level(logging.WARNING):
        extracted_data = extract_products(TARGET_URL)

    assert len(extracted_data) == 0
    assert "Tidak ada kartu produk yang ditemukan" in caplog.text

def test_extract_products_request_error(requests_mock, caplog):
    """Test penanganan error saat request (misal, 404 Not Found)."""
    requests_mock.get(TARGET_URL, status_code=404, reason="Not Found")

    with caplog.at_level(logging.ERROR):
        extracted_data = extract_products(TARGET_URL)

    assert len(extracted_data) == 0
    assert f"Error saat request ke {TARGET_URL}" in caplog.text
    assert "404 Client Error: Not Found" in caplog.text # Dari response.raise_for_status()

def test_extract_products_timeout(requests_mock, caplog):
    """Test penanganan error timeout."""
    requests_mock.get(TARGET_URL, exc=requests.exceptions.Timeout)

    with caplog.at_level(logging.ERROR):
        extracted_data = extract_products(TARGET_URL)

    assert len(extracted_data) == 0
    assert f"Request timeout saat mencoba mengakses {TARGET_URL}" in caplog.text

def test_extract_products_attribute_error(requests_mock, caplog):
    """Test jika struktur HTML berubah dan menyebabkan AttributeError."""
    # HTML dengan struktur berbeda (misal, tidak ada 'h3' untuk nama)
    MOCK_HTML_BAD_STRUCTURE = """
    <html><body>
      <div class="product-card">
        <a href="/product/t-shirt-keren">
          <div class="product-title"> Kaos Polos Keren  </div> <!-- Bukan h3 -->
        </a>
        <span class="product-price"> Rp 75.000 </span>
      </div>
    </body></html>
    """
    requests_mock.get(TARGET_URL, text=MOCK_HTML_BAD_STRUCTURE, status_code=200)

    with caplog.at_level(logging.WARNING):
        extracted_data = extract_products(TARGET_URL)

    # Harusnya skip produk yang error dan lanjut (atau return kosong jika hanya 1)
    # Jika extract.py kita menggunakan try-except di dalam loop:
    assert len(extracted_data) == 0
    assert "Gagal mengekstrak data dari satu kartu produk" in caplog.text
    assert "AttributeError" in caplog.text # Atau error spesifik lainnya