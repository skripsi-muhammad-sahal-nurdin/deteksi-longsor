from flask import Blueprint, current_app
from src.api.handler import predict_handler

# Buat Blueprint untuk API
api_blueprint = Blueprint('api', __name__)

@api_blueprint.route('/predict', methods=['POST'])
def predict():
    model = current_app.config['MODEL']
    upload_folder = current_app.config['UPLOAD_FOLDER']
    result_folder = current_app.config['RESULT_FOLDER']
    return predict_handler(model, upload_folder, result_folder)