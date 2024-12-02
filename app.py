from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load your trained model
model = load_model("D:\Important\Capstone Bangkit\AgriScan\backend\Flask_AgriScan\model\my_model.h5")

@app.route('/')
def home():
    return "Tomato Leaf Disease Classification API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if an image file is included in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Open and preprocess the image
        img = Image.open(file)
        img = img.resize((224, 224))  # Resize to match your model's input size
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize if your model expects normalized input
        
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1).tolist()
        
        return jsonify({
            "prediction": predicted_class,
            "confidence": prediction.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
