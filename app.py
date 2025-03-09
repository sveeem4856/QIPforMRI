# app.py
import os
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_PATH'] = 'brain_tumor_detection_model.h5'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the pre-trained model
def load_model():
    try:
        model = tf.keras.models.load_model(app.config['MODEL_PATH'])
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((150, 150))  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Make prediction
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    
    try:
        prediction = model.predict(img_array)
        probability = float(prediction[0][0])
        predicted_class = "Tumor" if probability > 0.5 else "No Tumor"
        return {
            "prediction": predicted_class,
            "probability": round(probability * 100 if predicted_class == "Tumor" else (1 - probability) * 100, 2)
        }
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Check if model is loaded
        global model
        if model is None:
            model = load_model()
            if model is None:
                return jsonify({"error": "Model could not be loaded"})
        
        # Make prediction
        result = predict_image(file_path)
        
        # Add file path for display
        result["image_path"] = filename
        
        return jsonify(result)
    
    return jsonify({"error": "File type not allowed"})

if __name__ == '__main__':
    app.run(debug=True)




import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('brain_tumor_detection_model.h5')

# Initialize Flask app
app = Flask(__name__)

# Set the path to save uploaded files
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Prediction function
def predict_tumor(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0  # Rescale pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    result = "Tumor" if prediction > 0.5 else "No Tumor"
    confidence = prediction[0][0] * 100 if result == "Tumor" else (1 - prediction[0][0]) * 100

    return result, confidence

# Route to upload and predict image
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if a file is part of the request
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        # If the user doesn't select a file, browser also submits an empty part without filename
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Get prediction
            result, confidence = predict_tumor(file_path)

            return render_template('result.html', result=result, confidence=confidence, filename=filename)

    return render_template('upload.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
