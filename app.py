from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import csv

app = Flask(__name__)
app.secret_key = "your_secret_key"
app.config["UPLOAD_FOLDER"] = os.path.join(app.root_path, "static/uploads")
app.config["PREDICTIONS_CSV"] = os.path.join(app.root_path, "predictions.csv")

# Ensure the upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load the model
model = load_model(r"C:/Users/diwan/Downloads/my_flask_app/CustomMod.keras")

# Class names
class_names = [
    "1080Lines", "1400Ripples", "Air_Compressor", "Blip", "Chirp",
    "Extremely_Loud", "Helix", "Koi_Fish", "Light_Modulation", 
    "Low_Frequency_Burst", "Low_Frequency_Lines", "No_Glitch", 
    "None_of_the_Above", "Paired_Doves", "Power_Line", 
    "Repeating_Blips", "Scattered_Light", "Scratchy", "Tomte", 
    "Violin_Mode", "Wandering_Line", "Whistle"
]

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(120, 142)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def save_predictions_to_csv(predictions):
    with open(app.config["PREDICTIONS_CSV"], mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Predicted Label", "Confidence Level"])
        for pred in predictions:
            writer.writerow([pred['image_name'], pred['name'], round(pred['confidence'] * 100, 2)])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "files" not in request.files:
            flash("No file part")
            return redirect(request.url)

        files = request.files.getlist("files")
        predictions = []

        for file in files:
            if file.filename == "":
                flash("No selected file")
                continue

            if not allowed_file(file.filename):
                flash(f"File '{file.filename}' is not a valid image. Please upload a valid image file.")
                continue
            
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            img_array = preprocess_image(file_path)
            pred = model.predict(img_array)[0]
            top_index = np.argmax(pred)
            
            predictions.append({
                "image_name": file.filename,
                "name": class_names[top_index],
                "confidence": pred[top_index],
                "image_url": url_for('uploaded_file', filename=file.filename)
            })

        save_predictions_to_csv(predictions)
        
        return render_template("index.html", predictions=predictions)

    return render_template("index.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route('/download_predictions')
def download_predictions():
    return send_from_directory(app.root_path, "predictions.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
