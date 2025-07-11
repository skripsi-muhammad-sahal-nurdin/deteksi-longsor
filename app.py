import os
from flask import Flask, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from src.services.model_loader import load_keras_model
from src.api.routes import api_blueprint

# Muat environment variables dari file .env
load_dotenv() 

app = Flask(__name__)
CORS(app)

# Konfigurasi Folder
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['RESULT_FOLDER'] = 'results/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Muat Model dari GCS (menggunakan URL dari .env)
MODEL_URL = os.getenv('MODEL_URL') # <-- Tambahkan ini
if not MODEL_URL:
    raise ValueError("MODEL_URL tidak ditemukan di environment variable. Harap set di file .env")

app.config['MODEL'] = load_keras_model(MODEL_URL) 

# Daftarkan Blueprint
app.register_blueprint(api_blueprint)

# Rute untuk menyajikan gambar hasil deteksi
@app.route('/results/<filename>')
def serve_result_image(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    # Untuk development, jalankan seperti ini
    app.run(host='0.0.0.0', port=5000, debug=True)
    # Untuk production, gunakan gunicorn: gunicorn --bind 0.0.0.0:8080 app:app