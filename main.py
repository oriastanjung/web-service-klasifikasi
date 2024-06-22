from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import base64
import os
from flask import Blueprint, send_from_directory
from controllerKlasifikasi import klasifikasiMangrove
import numpy as np
import cv2

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Menambahkan ini untuk mengaktifkan CORS


# Buat blueprint untuk file statis
static_bp = Blueprint('static', __name__, static_folder='gambar_data_tanaman')

# Tambahkan route untuk menyajikan gambar
@static_bp.route('/gambar_data_tanaman/<path:filename>')
def serve_image(filename):
    return send_from_directory(static_bp.static_folder, filename)

# Daftarkan blueprint ke aplikasi Flask
app.register_blueprint(static_bp)

@app.route('/', methods=['GET'])
def index():
    return jsonify("hello world")

@app.route('/mangrove/get-data', methods=['GET'])
def get_data():
    with open('dataTanaman.json', 'r') as file:
        data = json.load(file)
    return jsonify(data)

@app.route('/mangrove/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.json:
        return jsonify({"error": "No image part in the request"}), 400

    image_b64 = request.json['image']
    # Decode base64 to image data
    image_data = base64.b64decode(image_b64)

    # Convert image data to OpenCV format
    nparr = np.frombuffer(image_data, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Perform classification
    klasifikasi_mangrove, id_mangrove = klasifikasiMangrove(img_cv)

    # Load dataTanaman.json
    with open('dataTanaman.json', 'r') as file:
        data_tanaman = json.load(file)

    # Find data by ID
    result_data = None
    for data in data_tanaman:
        if data['id'] == id_mangrove+1:
            result_data = data
            break

    if result_data:
        return jsonify({
            "message": "Sukses Klasifikasi Mangrove!",
            "result": str(klasifikasi_mangrove),
            "id": str(id_mangrove +1),
            "data_tanaman": result_data
        })
    else:
        return jsonify({
            "message": "Sukses Klasifikasi Mangrove!",
            "result": str(klasifikasi_mangrove),
            "id": str(id_mangrove),
            "data_tanaman": "Data not found for ID " + str(id_mangrove + 1)
        })
        
        
        
        
        
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=7860,debug=True)
    # app.run(debug=True)
