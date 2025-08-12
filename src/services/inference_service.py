import h5py
import numpy as np
from scipy.ndimage import label, binary_opening # Pastikan binary_opening di-import

# --- FUNGSI-FUNGSI ANALISIS (SAMA SEPERTI DI NOTEBOOK) ---

def analyze_landslide_mask(prediction_mask, pixel_area_sqm=100.0):
    """Menganalisis mask untuk menghitung jumlah dan luas longsor."""
    structure = np.ones((3, 3), dtype=np.int32)
    labeled_mask, num_features = label(prediction_mask, structure=structure)
    if num_features == 0:
        return {'total_landslides': 0, 'landslides_details': []}, labeled_mask
    
    pixel_counts = np.bincount(labeled_mask.ravel())[1:]
    analysis_result = {'total_landslides': num_features, 'landslides_details': []}
    
    for i, count in enumerate(pixel_counts):
        analysis_result['landslides_details'].append({
            'id': i + 1,
            'area_in_pixels': int(count),
            'area_in_sqm': float(count * pixel_area_sqm)
        })
    return analysis_result, labeled_mask

# --- [PERBAIKAN] FUNGSI filter_predictions SEKARANG LENGKAP ---
def filter_predictions(stats_dict, labeled_mask, min_area_sqm):
    """Menyaring deteksi longsor dan merekonstruksi mask yang bersih."""
    # Saring daftar 'landslides_details'
    filtered_details = [
        detail for detail in stats_dict['landslides_details'] 
        if detail['area_in_sqm'] >= min_area_sqm
    ]
    
    # Jika tidak ada yang tersisa setelah disaring
    if not filtered_details:
        return {'total_landslides': 0, 'landslides_details': []}, np.zeros_like(labeled_mask)

    # Buat mask baru yang bersih (awalnya semua hitam/0)
    clean_mask = np.zeros_like(labeled_mask, dtype=np.uint8)
    
    # Dapatkan ID dari longsor yang lolos saringan
    valid_ids = {detail['id'] for detail in filtered_details}

    # "Gambar ulang" hanya longsor yang valid ke mask yang bersih
    for i in valid_ids:
        clean_mask[labeled_mask == i] = 1

    # Buat dictionary hasil akhir yang sudah bersih
    final_stats = {
        'total_landslides': len(filtered_details),
        'landslides_details': filtered_details
    }
    
    # Kembalikan statistik DAN mask yang sudah bersih
    return final_stats, clean_mask

# --- FUNGSI UTAMA UNTUK INFERENSI (SUDAH DIMODIFIKASI) ---

def preprocess_and_predict(model, file_path):
    """Melakukan preprocessing, prediksi, dan analisis lengkap dengan post-processing."""
    
    # =================================================================
    # [PERBAIKAN] Definisikan best_threshold di sini sebagai konstanta.
    # Nilai ini diambil dari hasil analisis pada data validasi di notebook.
    # Ganti angka ini jika mendapatkan nilai threshold baru dari eksperimen.
    # =================================================================
    BEST_THRESHOLD = 0.3519 
    # =================================================================
    
    # 1. Preprocessing (Tidak ada perubahan)
    with h5py.File(file_path, 'r') as hdf:
        data = np.array(hdf.get('img'))
    
    processed_data = np.zeros((128, 128, 6))
    data[np.isnan(data)] = 1e-6
    mid_rgb = data[:, :, 1:4].max() / 2.0 if data[:, :, 1:4].max() > 0 else 1.0
    mid_slope = data[:, :, 12].max() / 2.0 if data[:, :, 12].max() > 0 else 1.0
    mid_elevation = data[:, :, 13].max() / 2.0 if data[:, :, 13].max() > 0 else 1.0
    
    data_red = data[:, :, 3]
    data_nir = data[:, :, 7]
    denominator = np.add(data_nir, data_red)
    denominator[denominator == 0] = 1e-6
    data_ndvi = np.divide(data_nir - data_red, denominator)
    
    processed_data[:, :, 0] = 1 - data[:, :, 3] / mid_rgb
    processed_data[:, :, 1] = 1 - data[:, :, 2] / mid_rgb
    processed_data[:, :, 2] = 1 - data[:, :, 1] / mid_rgb
    processed_data[:, :, 3] = data_ndvi
    processed_data[:, :, 4] = 1 - data[:, :, 12] / mid_slope
    processed_data[:, :, 5] = 1 - data[:, :, 13] / mid_elevation
    processed_data[np.isnan(processed_data)] = 1e-6
    
    original_rgb = data[:, :, 3:0:-1]

    # 2. Prediksi
    input_batch = np.expand_dims(processed_data, axis=0)
    pred_mask = model.predict(input_batch)[0]
    
    # [PERBAIKAN] Gunakan konstanta yang sudah didefinisikan
    pred_mask_binary = (pred_mask > BEST_THRESHOLD).astype(np.uint8)[:, :, 0]
    
    # 3. Post-Processing dan Analisis
    structure = np.ones((2, 2))
    MINIMUM_AREA_SQM = 100
    
    # Langkah 3.1: Bersihkan noise kecil
    cleaned_mask = binary_opening(pred_mask_binary, structure=structure).astype(np.uint8)
    
    # Langkah 3.2: Lakukan analisis awal pada mask yang sudah bersih
    raw_stats, labeled_mask = analyze_landslide_mask(cleaned_mask)
    
    # Langkah 3.3: Saring hasil berdasarkan luas minimum
    # [PERBAIKAN] Tangkap kedua output: statistik DAN mask final
    final_landslide_stats, final_mask = filter_predictions(raw_stats, labeled_mask, min_area_sqm=MINIMUM_AREA_SQM)
    
    # [PERBAIKAN] Kembalikan mask FINAL yang sudah bersih dan tersaring
    return original_rgb, final_mask, final_landslide_stats