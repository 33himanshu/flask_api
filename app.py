from flask import Flask, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
import os

app = Flask(__name__)

# Load the pre-trained model
model = load_model('braintumor.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (150, 150))
        img_array = np.array(img).astype('float32')
        img_array = img_array.reshape(1, 150, 150, 3)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
        predicted_label = labels[predicted_class]

        return jsonify({'prediction': predicted_label})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Get the port from environment variable or default to 5000
port = int(os.environ.get("PORT", 5000))

if __name__ == '__main__':
    # Run the app on the specified port and host
    app.run(host='0.0.0.0', port=port)
