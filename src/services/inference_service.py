import h5py
import numpy as np
from scipy.ndimage import label

def preprocess_and_predict(model, file_path):
    """Melakukan preprocessing, prediksi, dan analisis."""
    # 1. Preprocessing
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
    pred_mask_binary = (pred_mask > 0.5).astype(np.uint8)[:, :, 0]
    
    # 3. Analisis
    landslide_stats = analyze_landslide_mask(pred_mask_binary)
    
    return original_rgb, pred_mask_binary, landslide_stats

def analyze_landslide_mask(prediction_mask, pixel_area_sqm=100.0):
    """Menganalisis mask untuk menghitung jumlah dan luas longsor."""
    structure = np.ones((3, 3), dtype=np.int32)
    labeled_mask, num_features = label(prediction_mask, structure=structure)
    if num_features == 0:
        return {'total_landslides': 0, 'landslides_details': []}
    
    pixel_counts = np.bincount(labeled_mask.ravel())[1:]
    analysis_result = {'total_landslides': num_features, 'landslides_details': []}
    
    for i, count in enumerate(pixel_counts):
        analysis_result['landslides_details'].append({
            'id': i + 1,
            'area_in_pixels': int(count),
            'area_in_sqm': float(count * pixel_area_sqm)
        })
    return analysis_result