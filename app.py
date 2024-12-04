# from flask import Flask, request, jsonify
# import cv2
# import numpy as np
# from keras.models import load_model

# app = Flask(__name__)

# model = load_model('braintumor.h5')

# @app.route('/predict', methods=['POST'])
# def predict():
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
#     if file:
#         try:
#             # Read and decode the image
#             img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
#             # Resize the image to match the input size expected by the model
#             img = cv2.resize(img, (150, 150))
#             # Convert the image to a numpy array
#             img_array = np.array(img)
#             # Reshape the array to match the input shape expected by the model
#             img_array = img_array.reshape(1, 150, 150, 3)
#             # Make predictions
#             predictions = model.predict(img_array)
#             # Get the index of the highest probability prediction
#             predicted_class = np.argmax(predictions, axis=1)
#             # Define the label mapping
#             labels = ['glioma','meningioma', 'notumor', 'pituitary']
#             # Map the index to the corresponding label
#             predicted_label = labels[predicted_class[0]]
#             return jsonify({'prediction': predicted_label})
#         except Exception as e:
#             return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

model = load_model('braintumor.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        try:
            img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (150, 150))
            img_array = np.array(img)
            img_array = img_array.reshape(1, 150, 150, 3)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            labels = ['glioma','meningioma', 'notumor', 'pituitary']
            predicted_label = labels[predicted_class]
            return jsonify({'prediction': predicted_label})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


# from keras.models import load_model
# import cv2
# import numpy as np

# # Load the model
# model = load_model('braintumor.h5')

# # Define the labels
# labels = ['glioma','meningioma', 'notumor', 'pituitary']

# # Load and preprocess the new image
# image_path = r"C:\Users\HIMANSHU\Downloads\archive (1)\Testing\meningioma\Te-me_0294.jpg" 
# img = cv2.imread(image_path)
# img = cv2.resize(img, (150, 150))  # Resize to match the input size expected by the model
# img_array = np.expand_dims(img, axis=0)  # Add batch dimension

# # Make predictions
# predictions = model.predict(img_array)

# # Get the index of the highest probability prediction
# predicted_class = np.argmax(predictions, axis=1)[0]

# # Map the index to the corresponding label
# predicted_label = labels[predicted_class]

# print(f'Predicted label: {predicted_label}')