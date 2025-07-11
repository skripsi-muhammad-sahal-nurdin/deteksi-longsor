import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from google.cloud import storage

def create_and_upload_image(original_rgb, prediction_mask, result_image_name, result_folder):
    """
    Membuat gambar deteksi, meng-upload ke GCS, dan mengembalikan URL publik.
    """
    # Pastikan folder lokal ada untuk penyimpanan sementara
    os.makedirs(result_folder, exist_ok=True)
    local_path = os.path.join(result_folder, result_image_name)
    
    # 1. Buat dan simpan gambar secara lokal (seperti sebelumnya)
    plt.figure(figsize=(5, 5), frameon=False)
    if original_rgb.max() > 0:
        display_rgb = original_rgb / original_rgb.max()
    else:
        display_rgb = original_rgb
    plt.imshow(display_rgb)
    plt.imshow(prediction_mask, cmap='Reds', alpha=0.5)
    plt.axis('off')
    plt.savefig(local_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # 2. Upload file dari path lokal ke GCS
    try:
        bucket_name = os.getenv('GCS_BUCKET_NAME')
        if not bucket_name:
            print("[!] GCS_BUCKET_NAME tidak diatur. Melewatkan upload.")
            return f"/results/{result_image_name}" # Fallback ke URL lokal

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(result_image_name)

        blob.upload_from_filename(local_path)
        
        # 3. Jadikan file dapat diakses publik
        blob.make_public()

        print(f"[*] Gambar {result_image_name} berhasil diupload ke {blob.public_url}")
        return blob.public_url # Kembalikan URL publik

    except Exception as e:
        print(f"[!] Gagal upload ke GCS: {e}")
        return f"/results/{result_image_name}" # Fallback ke URL lokal
    finally:
        # 4. Hapus file lokal setelah diupload
        if os.path.exists(local_path):
            os.remove(local_path)