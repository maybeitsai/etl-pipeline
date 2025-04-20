# tests/test_transform.py

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from datetime import datetime
import numpy as np

# Pastikan path ke utils benar
from utils.transform import transform_data

@pytest.fixture
def sample_raw_data():
    """Fixture untuk menyediakan data mentah sampel."""
    return [
        {'name': '  Kaos Polos Keren ', 'price_raw': 'Rp 75.000', 'category': 'T-Shirt', 'url': '/produk/kaos1'},
        {'name': 'Jaket Bomber Stylish', 'price_raw': 'Rp 250.000', 'category': 'JACKET', 'url': '/produk/jaket1'},
        {'name': 'Celana Chino Nyaman', 'price_raw': 'Rp 180000', 'category': 'Pants ', 'url': '/produk/celana1'}, # Harga tanpa format
        {'name': 'Produk Error Harga', 'price_raw': 'Harga Tidak Valid', 'category': 'Unknown', 'url': '/produk/error'},
        {'name': 'Produk Tanpa Harga', 'category': 'Outerwear', 'url': '/produk/outer1'}, # Key 'price_raw' hilang
        {} # Data kosong
    ]

@pytest.fixture
def expected_transformed_data():
    """Fixture untuk menyediakan data hasil transformasi yang diharapkan."""
    # Data ini HARUS sesuai dengan logika di transform_data
    # Kolom 'scraped_at' akan dibandingkan secara terpisah karena nilainya dinamis
    return pd.DataFrame({
        'name': ['Kaos Polos Keren', 'Jaket Bomber Stylish', 'Celana Chino Nyaman', 'Produk Error Harga', 'Produk Tanpa Harga'],
        'category': ['t-shirt', 'jacket', 'pants', 'unknown', 'outerwear'],
        'price': [75000.0, 250000.0, 180000.0, 0.0, 0.0], # Asumsi error/missing price jadi 0
        'url': ['/produk/kaos1', '/produk/jaket1', '/produk/celana1', '/produk/error', '/produk/outer1'],
        # scraped_at ditambahkan di test
    })


def test_transform_data_success(sample_raw_data, expected_transformed_data):
    """Test transformasi data berhasil."""
    # Hapus data kosong dari raw data karena transform akan skip
    valid_raw_data = [d for d in sample_raw_data if d]
    # Hapus data tanpa price_raw karena transform akan error jika tidak dihandle
    # Asumsikan transform_data kita bisa handle KeyError price_raw dan tetap jalan
    # Mari kita modifikasi transform_data sedikit untuk menangani KeyError price_raw
    # Di transform_data, dalam blok try harga:
    # try:
    #    df['price'] = df['price_raw'].astype(str)...
    # except KeyError:
    #    logging.warning("Kolom 'price_raw' tidak ditemukan, mengisi harga dengan 0.")
    #    df['price'] = 0 # Default value

    # Filter data raw yang punya 'price_raw' atau yang akan diisi 0
    # Baris ke-5 ('Produk Tanpa Harga') akan punya price=0
    # Baris ke-4 ('Produk Error Harga') akan punya price=0
    # Baris terakhir ({}) akan difilter

    transformed_df = transform_data(valid_raw_data)

    assert not transformed_df.empty
    # Kolom 'scraped_at' ada dan tipenya datetime
    assert 'scraped_at' in transformed_df.columns
    assert pd.api.types.is_datetime64_any_dtype(transformed_df['scraped_at'])

    # Bandingkan kolom lain (tanpa scraped_at)
    cols_to_compare = expected_transformed_data.columns.tolist()
    # Gunakan check_dtype=False karena price bisa jadi int vs float (75000.0 vs 75000)
    # Gunakan check_like=True untuk mengabaikan urutan baris jika diperlukan
    assert_frame_equal(
        transformed_df[cols_to_compare].sort_values(by='name').reset_index(drop=True),
        expected_transformed_data.sort_values(by='name').reset_index(drop=True),
        check_dtype=False,
        # atol=0.1 # Toleransi untuk float jika perlu
    )

def test_transform_data_empty_input():
    """Test transformasi dengan input list kosong."""
    transformed_df = transform_data([])
    assert transformed_df.empty
    assert isinstance(transformed_df, pd.DataFrame)

def test_transform_data_missing_columns(caplog):
    """Test jika kolom penting selain price_raw hilang (misal 'name')."""
    raw_data = [
        {'price_raw': 'Rp 50.000', 'category': 'Test', 'url': '/test'}
        # Tidak ada 'name'
    ]
    with caplog.at_level(logging.WARNING):
         transformed_df = transform_data(raw_data)

    # Seharusnya tetap memproses kolom lain dan kolom 'name' tidak ada
    assert 'name' not in transformed_df.columns
    assert 'Kolom \'name\' tidak ditemukan.' in caplog.text
    assert not transformed_df.empty # Tetap menghasilkan data
    assert transformed_df.iloc[0]['price'] == 50000.0

def test_transform_price_cleaning():
    """Fokus pada berbagai format harga."""
    raw = [
        {'name':'p1', 'price_raw':'Rp 1.234.567', 'cat':'c1', 'url':'u1'},
        {'name':'p2', 'price_raw':'Rp100000', 'cat':'c1', 'url':'u2'},
        {'name':'p3', 'price_raw':'  50.000  ', 'cat':'c1', 'url':'u3'},
        # Tambahkan format lain jika ada (misal, pakai koma desimal)
        # {'name':'p4', 'price_raw':'Rp 12,500.99', 'cat':'c1', 'url':'u4'} # Perlu regex berbeda
    ]
    df = transform_data(raw)
    expected_prices = [1234567.0, 100000.0, 50000.0]
    assert df['price'].tolist() == expected_prices