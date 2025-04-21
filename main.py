from flask import Flask, request, jsonify
import io
from PIL import Image
from ultralytics import YOLO
import base64
import numpy as np
import os
import time

app = Flask(__name__)

model = YOLO('./best_ncnn_model')

# Define your class names based on the model training
class_names = ['EH', 'IB', 'R', 'U']  # Example: Entry Hole, Insect Bite, Rot, Unripe

@app.route('/')
def home():
    return "Hello, Flask API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    # Decode base64 image
    image_data = base64.b64decode(data['image'])
    img = Image.open(io.BytesIO(image_data))

    # Run model prediction
    results = model(img)
    result = results[0]

    # Extract boxes, confidence scores, and class ids
    boxes = result.boxes.xyxy.cpu().numpy().tolist()
    conf = result.boxes.conf.cpu().numpy().tolist()
    cls = result.boxes.cls.cpu().numpy().tolist()

    class_counts = {class_name: 0 for class_name in class_names}

    for _, _, cl in zip(boxes, conf, cls):
        class_id = int(cl)
        class_name = class_names[class_id] if class_id < len(class_names) else f"Unknown({class_id})"
        class_counts[class_name] += 1

    # Generate unique filename using epoch time in milliseconds
    timestamp = int(time.time() * 1000)
    filename = f"{timestamp}.jpg"

    # Save the image with predictions
    result.save(filename=filename)

    # Re-encode processed image
    with open(filename, 'rb') as f:
        processed_image_base64 = base64.b64encode(f.read()).decode('utf-8')

    # Delete the file after encoding
    if os.path.exists(filename):
        os.remove(filename)

    return jsonify({
        'timestamp': timestamp,
        'class_counts': class_counts,
        'processed_image': processed_image_base64
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)