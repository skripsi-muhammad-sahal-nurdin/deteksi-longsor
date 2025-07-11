import os
import uuid
from datetime import datetime, timezone
from flask import request, jsonify
from werkzeug.utils import secure_filename
from src.services.inference_service import preprocess_and_predict
from src.services.firestore_service import save_prediction
# from src.utils.image_generator import create_detection_image
from src.services.storage_service import create_and_upload_image

def predict_handler(model, upload_folder, result_folder):
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong.'}), 400

    if file and file.filename.endswith('.h5'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)

        try:
            original_rgb, pred_mask_binary, stats = preprocess_and_predict(model, filepath)
            
            result_image_name = f"result_{filename.replace('.h5', '.png')}"
            
            # 2. BUAT DAN UPLOAD GAMBAR KE GCS
            detection_image_url = create_and_upload_image(
                original_rgb, 
                pred_mask_binary, 
                result_image_name,
                result_folder
            ) # <-- PANGGIL FUNGSI BARU

            is_landslide = stats['total_landslides'] > 0
            prediction_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc).isoformat()
            
            response_data = {
                'success': True,
                'filename': filename,
                'landslide_detected': is_landslide,
                'message': 'Tanah longsor terdeteksi.' if is_landslide else 'Tidak ada tanah longsor yang terdeteksi.',
                'frequency': stats['total_landslides'],
                'details': stats['landslides_details'],
                'detection_image_url': detection_image_url # <-- GUNAKAN URL DARI GCS
            }

            
            firestore_data = {
                'id': prediction_id,
                'createdAt': timestamp,
                **response_data
            }
            save_prediction(prediction_id, firestore_data)
            return jsonify(response_data), 200

        except Exception as e:
            return jsonify({'error': f'Terjadi kesalahan saat pemrosesan: {str(e)}'}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return jsonify({'error': 'Format file tidak didukung. Harap unggah file .h5.'}), 400