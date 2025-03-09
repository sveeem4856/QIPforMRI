from flask import Flask, render_template, request
import os
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# # # Load the pre-trained model
model = load_model('tumor_classifier.h5')

from flask import Flask, render_template, request, redirect, url_for, send_file
import requests

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"  # Folder to store images
REMOVE_BG_API_KEY = "PTrRq1gTAn1vRzqsMhRgxqY8"  # Replace with your API key

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "POST":
        # name = request.form.get("name")
        # age = request.form.get("age")
        # q3 = request.form.get("q3")
        # q4 = request.form.get("q4")
        # q5 = request.form.get("q5")
        # q6 = request.form.get("q6")


        file = request.files["file"]
        if file:
            input_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(input_path)  # Save the original image

            # Send to Remove.bg API
            with open(input_path, "rb") as image_file:
                response = requests.post(
                    "https://api.remove.bg/v1.0/removebg",
                    files={"image_file": image_file},
                    data={"size": "auto"},
                    headers={"X-Api-Key": REMOVE_BG_API_KEY},
                )

            if response.status_code == 200:
                output_path = os.path.join(app.config["UPLOAD_FOLDER"], f"no-bg-{file.filename}")
                with open(output_path, "wb") as out_file:
                    out_file.write(response.content)

                # Predict tumor type using the processed image
                label, confidence = predict_image(output_path)

                # Redirect to results page with processed image
                return render_template("result.html", image_url=output_path, label=label, confidence=confidence)
            else:
                return "Error removing background", 400

    return render_template("main.html")
def predict_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0  # Rescale pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    result = "Tumor" if prediction > 0.5 else "No Tumor"
    confidence = prediction[0][0] * 100 if result == "Tumor" else (1 - prediction[0][0]) * 100

    return result, confidence
@app.route("/download/<filename>")
def download_file(filename):
    return send_file(os.path.join(app.config["UPLOAD_FOLDER"], filename), as_attachment=True)


@app.route('/home')
def home():
    return render_template('home.html')
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
