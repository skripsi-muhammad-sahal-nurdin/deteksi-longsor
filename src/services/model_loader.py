# src/services/model_loader.py

import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from google.cloud import storage

# --- Definisi custom metrics tetap sama ---
def precision_m(y_true, y_pred):
    # ... (kode tidak berubah)
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def recall_m(y_true, y_pred):
    # ... (kode tidak berubah)
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall

def f1_m(y_true, y_pred):
    # ... (kode tidak berubah)
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+tf.keras.backend.epsilon()))
# -----------------------------------------

def load_keras_model(model_url):
    """
    Mengunduh model dari GCS ke file sementara, lalu memuatnya.
    """
    temp_model_path = "/tmp/model.keras" # Lokasi sementara di server
    
    try:
        # 1. Parsing GCS URL
        if not model_url.startswith("gs://"):
            raise ValueError("URL Model harus dalam format 'gs://bucket-name/path/to/model.keras'")
        
        bucket_name = model_url.split("/")[2]
        source_blob_name = "/".join(model_url.split("/")[3:])

        # 2. Unduh model dari GCS
        print(f"[*] Mengunduh model dari gs://{bucket_name}/{source_blob_name}...")
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(temp_model_path)
        print("[*] Model berhasil diunduh ke lokasi sementara.")

        # 3. Muat model dari file lokal sementara
        print(f"[*] Memuat model dari {temp_model_path}...")
        custom_objects = {'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m}
        model = load_model(temp_model_path, custom_objects=custom_objects)
        print("[*] Model berhasil dimuat.")
        return model

    except Exception as e:
        print(f"[!] Error saat memuat model dari GCS: {e}")
        return None
        
    finally:
        # 4. Hapus file sementara setelah model dimuat (atau jika terjadi error)
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
            print("[*] File model sementara telah dihapus.")