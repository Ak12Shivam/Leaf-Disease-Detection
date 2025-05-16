from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import cv2
from keras.models import load_model
import tensorflow as tf
from io import BytesIO
import os
import logging

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})  # Restrict CORS to /predict endpoint

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Loading the Model
try:
    model = load_model('plant_disease_model.h5')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Name of Classes
CLASS_NAMES = ('Tomato-Bacterial_spot', 'Potato-Early_blight', 'Corn-Common_rust')

# Route to serve the index.html
@app.route('/')
def serve_index():
    try:
        return send_file('index.html')
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        return jsonify({'error': 'Failed to serve frontend'}), 500

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict_disease():
    logger.debug("Received prediction request")
    
    if 'plant_image' not in request.files:
        logger.error("No image provided in request")
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['plant_image']
    
    # Validate file type
    allowed_types = ['image/jpeg', 'image/png']
    if not file.content_type in allowed_types:
        logger.error(f"Invalid file type: {file.content_type}")
        return jsonify({'error': 'Only JPG and PNG images are supported'}), 400
    
    try:
        # Read and decode image
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if opencv_image is None:
            logger.error("Failed to decode image")
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Preprocess image
        opencv_image = cv2.resize(opencv_image, (256, 256))
        opencv_image = opencv_image.reshape(1, 256, 256, 3).astype(np.float32) / 255.0  # Normalize image
        
        # Make prediction
        Y_pred = model.predict(opencv_image)
        predicted_class = np.argmax(Y_pred, axis=1)[0]
        result = CLASS_NAMES[predicted_class]
        confidences = Y_pred[0].tolist()  # Convert to list for JSON response
        
        logger.info(f"Prediction successful: {result}")
        return jsonify({
            'result': result,
            'confidences': confidences
        })
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)