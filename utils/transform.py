import pandas as pd
from datetime import datetime
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def transform_data(raw_data: list[dict]) -> pd.DataFrame:
    """
    Membersihkan dan mentransformasi data produk mentah.

    Args:
        raw_data (list[dict]): List dictionary data produk mentah.

    Returns:
        pd.DataFrame: DataFrame berisi data produk yang sudah bersih.
                      Mengembalikan DataFrame kosong jika input kosong atau error.
    """
    if not raw_data:
        logging.warning("Menerima data mentah kosong untuk transformasi.")
        return pd.DataFrame()

    df = pd.DataFrame(raw_data)
    logging.info(f"Memulai transformasi untuk {len(df)} baris data.")

    # 1. Transformasi Harga (Contoh: "Rp 150.000" -> 150000.0)
    try:
        # Hapus 'Rp', '.', dan spasi, lalu konversi ke float
        # Perlu sangat hati-hati dengan format harga aktual di website
        df['price'] = df['price_raw'].astype(str).apply(lambda x: re.sub(r'[Rp.\s]', '', x))
        # Jika ada koma sebagai desimal, ganti dengan titik (sesuaikan jika perlu)
        # df['price'] = df['price'].str.replace(',', '.', regex=False)
        df['price'] = pd.to_numeric(df['price'], errors='coerce') # Gagal konversi jadi NaN
        # Isi NaN dengan 0 atau nilai default lain, atau hapus baris
        df['price'].fillna(0, inplace=True)
        logging.info("Kolom harga berhasil ditransformasi.")
    except KeyError:
        logging.error("Kolom 'price_raw' tidak ditemukan dalam data mentah.")
        # Mungkin return df kosong atau handle sesuai kebutuhan
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error saat transformasi harga: {e}")
        # Beri nilai default atau handle error
        df['price'] = 0

    # 2. Transformasi Nama
    try:
        df['name'] = df['name'].str.strip()
        logging.info("Kolom nama dibersihkan (strip whitespace).")
    except KeyError:
        logging.warning("Kolom 'name' tidak ditemukan.")
    except Exception as e:
        logging.error(f"Error saat transformasi nama: {e}")


    # 3. Transformasi Kategori (Contoh: Standardisasi)
    try:
        df['category'] = df['category'].str.strip().str.lower()
        # Logika tambahan jika perlu (misal mapping)
        logging.info("Kolom kategori dibersihkan dan diubah ke lowercase.")
    except KeyError:
        logging.warning("Kolom 'category' tidak ditemukan.")
    except Exception as e:
        logging.error(f"Error saat transformasi kategori: {e}")


    # 4. Tambah Kolom Timestamp
    df['scraped_at'] = datetime.now()

    # 5. Pilih dan Urutkan Kolom (sesuaikan dengan kebutuhan tabel DB/Sheets)
    # Pastikan kolom ini ada setelah transformasi
    final_columns = ['name', 'category', 'price', 'url', 'scraped_at']
    # Filter hanya kolom yang ada di DataFrame untuk menghindari KeyError
    existing_columns = [col for col in final_columns if col in df.columns]
    df_transformed = df[existing_columns]

    logging.info(f"Transformasi selesai. Menghasilkan {len(df_transformed)} baris data.")
    return df_transformed