from quart import Quart, request, jsonify, send_from_directory, Blueprint
from quart_cors import cors
import json
import base64
# import os
import numpy as np
import cv2
import asyncio  
import aiofiles
from controllerKlasifikasi import klasifikasiMangrove

app = Quart(__name__)
cors(app, allow_origin="*")  # Mengaktifkan CORS

# Buat blueprint untuk file statis
static_bp = Blueprint('static', __name__, static_folder='gambar_data_tanaman')

# Tambahkan route untuk menyajikan gambar
@static_bp.route('/gambar_data_tanaman/<path:filename>')
async def serve_image(filename):
    return await send_from_directory(static_bp.static_folder, filename)

# Daftarkan blueprint ke aplikasi Quart
app.register_blueprint(static_bp)

@app.route('/', methods=['GET'])
async def index():
    return jsonify("hello world")

@app.route('/mangrove/get-data', methods=['GET'])
async def get_data():
    async with aiofiles.open('dataTanaman.json', 'r') as file:
        data = await file.read()
    return jsonify(json.loads(data))

@app.route('/mangrove/upload-image', methods=['POST'])
async def upload_image():
    request_json = await request.json
    if 'image' not in request_json:
        return jsonify({"error": "No image part in the request"}), 400

    image_b64 = request_json['image']
    # Decode base64 to image data
    image_data = base64.b64decode(image_b64)

    # Convert image data to OpenCV format
    nparr = np.frombuffer(image_data, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Save the image locally
    # image_path = 'uploaded_image.png'
    # cv2.imwrite(image_path, img_cv)
    # Perform classification (synchronously)
    klasifikasi_mangrove, id_mangrove = await asyncio.to_thread(klasifikasiMangrove, img_cv)

    # Load dataTanaman.json
    async with aiofiles.open('dataTanaman.json', 'r') as file:
        data_tanaman_json = await file.read()
        data_tanaman = json.loads(data_tanaman_json)

    # Find data by ID
    result_data = None
    for data in data_tanaman:
        if data['id'] == id_mangrove + 1:
            result_data = data
            break

    if result_data:
        return jsonify({
            "message": "Sukses Klasifikasi Mangrove!",
            "result": str(klasifikasi_mangrove),
            "id": str(id_mangrove + 1),
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
    app.run(host='0.0.0.0', debug=True, port=5000)
