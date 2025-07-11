from google.cloud import firestore

def save_prediction(prediction_id, data):
    """Menyimpan data prediksi ke Firestore."""
    try:
        # Inisialisasi koneksi ke Firestore
        # Library akan otomatis menggunakan kredensial dari environment variable
        db = firestore.Client()
        
        # Tentukan koleksi dan ID dokumen
        doc_ref = db.collection('predictions').document(prediction_id)
        
        # Simpan data
        doc_ref.set(data)
        print(f"[*] Data prediksi {prediction_id} berhasil disimpan ke Firestore.")
        
    except Exception as e:
        print(f"[!] Gagal menyimpan ke Firestore: {e}")