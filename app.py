import numpy as np

from tensorflow.keras.models import load_model
from PIL import Image
import cv2
from flask import Flask, render_template, request, jsonify
import json
import base64

from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MODEL_FILE'] = 'eye_disease_model.h5'
app.config['LABELS_FILE'] = 'labels.txt'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


model = load_model(app.config['MODEL_FILE'], compile=False)
with open(app.config['LABELS_FILE'], 'r') as file:
    labels = file.read().splitlines()

def k_means_segmentation(image, k=3):

    # Calculate the histogram of the input image
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # Calculate the cumulative distribution function (CDF) of the histogram
    cdf = hist.cumsum()

    # Normalize the CDF to the range [0, 255]
    cdf_normalized = (cdf * 255) / cdf[-1]

    # Use the normalized CDF to perform histogram normalization
    normalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized)

    # Reshape the normalized image to its original shape
    normalized_image = normalized_image.reshape(image.shape)

    # Reshape the image to a 2D array of pixels
    pixels = normalized_image.reshape((-1, 1))

    # Convert pixel values to float
    pixels = np.float32(pixels)

    # Apply K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8-bit values
    centers = np.uint8(centers)

    # Map the labels to the centers
    segmented_image = centers[labels.flatten()]

    # Reshape the segmented image to the original shape
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image


def predict_eye_disease(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Preprocessing
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    segmented_image = k_means_segmentation(image_gray, k=7)
    resized_image = cv2.resize(segmented_image, (512, 512))
    
    # Expand dimensions to make it a batch of size 1
    data = np.expand_dims(resized_image, axis=0)
    
    # Make predictions
    predictions = model.predict(data)
    
    # Get the predicted class and confidence score
    index = np.argmax(predictions)
    class_name = labels[index]
    confidence_score = predictions[0][index]
    
    return class_name, confidence_score



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        # If the user does not select a file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file and allowed_file(file.filename):
            # Save the uploaded file to the upload folder
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Make predictions
            class_name, confidence_score = predict_eye_disease(file_path)

            # Render the prediction template
            return render_template('prediction.html', filename=filename, class_name=class_name, confidence=confidence_score)

    # Render the index template if the request method is not POST
    return render_template('index.html')

@app.route("/api/prediction", methods=["POST"])
def prediction_api():
    if request.method == "POST":
        # Check if the request contains JSON data
        if not request.is_json:
            return jsonify({
                "code": 400,
                "message": "Invalid JSON data"
            }), 400

        data = request.get_json()

        # Check if the JSON data contains the 'image' key
        if 'image' not in data:
            return jsonify({
                "code": 400,
                "message": "Missing 'image' key in JSON data"
            }), 400

        # Decode the base64-encoded image
        try:
            image_data = base64.b64decode(data['image'])
        except Exception as e:
            return jsonify({
                "code": 400,
                "message": str(e)
            }), 400

        # Save the decoded image to the upload folder
        filename = "uploaded_image.png"  # You can generate a unique filename here
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        with open(file_path, 'wb') as f:
            f.write(image_data)

        # Make predictions
        class_name, confidence_score = predict_eye_disease(file_path)

        # Return the prediction as JSON
        return jsonify({
            "status": {
                "code": 200,
                "message": "success predicting"
            },
            "data": {
                "filename": filename,
                "class_name": class_name,
                "confidence": confidence_score
            }
        })

    # Return an error if the request method is not POST
    return jsonify({
        "code": 405,
        "message": "Invalid request method"
    }), 405