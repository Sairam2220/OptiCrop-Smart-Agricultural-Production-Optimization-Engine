# ==========================================
# OptiCrop Smart Agricultural Production Optimization Engine
# Flask Web Application
# ==========================================

# Import required libraries
import numpy as np
import pickle
from flask import Flask, render_template, request

# Create Flask application
app = Flask(__name__)

# Load trained machine learning model
model = pickle.load(open("model/crop_model.pkl", "rb"))

# ==========================================
# Home Page Route
# ==========================================

@app.route("/")
def home():
    return render_template("home.html")

# ==========================================
# About Page Route
# ==========================================

@app.route("/about")
def about():
    return render_template("about.html")

# ==========================================
# Find Your Crop Page Route
# ==========================================

@app.route("/findyourcrop")
def findyourcrop():
    return render_template("findyourcrop.html")

# ==========================================
# Crop Prediction Route
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    N = float(request.form["N"])
    P = float(request.form["P"])
    K = float(request.form["K"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])

    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    prediction = model.predict(input_data)

    return render_template(
        "result.html",
        prediction_text=f"Recommended Crop: {prediction[0]}"
    )

# ==========================================
# Run Flask Application
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)