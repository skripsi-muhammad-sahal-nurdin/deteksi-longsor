import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

def create_detection_image(original_rgb, prediction_mask, output_path):
    """Menyimpan gambar RGB dengan overlay mask prediksi."""
    plt.figure(figsize=(5, 5), frameon=False)
    
    # Normalisasi RGB agar bisa ditampilkan dengan benar
    if original_rgb.max() > 0:
        display_rgb = original_rgb / original_rgb.max()
    else:
        display_rgb = original_rgb

    plt.imshow(display_rgb)
    # Overlay mask prediksi dengan warna merah transparan
    plt.imshow(prediction_mask, cmap='Reds', alpha=0.5)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close() # Penting untuk melepaskan memory