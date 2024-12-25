from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import re
import joblib
import tensorflow as tf
from ml_dl_fusion import get_result  # Import from the converted Python script

# Initialize Flask app
app = Flask(__name__)

# Load the saved models
knn_model = joblib.load('knn_model.pkl')  # Load KNN model
lstm_model = tf.keras.models.load_model('lstm_model.h5')  # Load LSTM model

# Define Flask routes
@app.route("/")
def home():
    return render_template("home.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    input_data = []
    result = None
    if request.method == 'POST':
        input_data.append(request.form.get("profilepic").lower())
        username = request.form.get("nctulr")
        nctulr = len(re.findall(r'\d+', username)) / len(username)
        input_data.append(nctulr)
        fullname = request.form.get("fnwc")
        fnwc = len(fullname.split())
        input_data.append(float(fnwc))
        nctfnlr = len(re.findall(r'\d+', fullname)) / len(fullname)
        input_data.append(float(nctfnlr))
        nmun = 1 if username == fullname else 0
        input_data.append(float(nmun))
        dsl = len(request.form.get("dsl"))
        input_data.append(float(dsl))
        input_data.append(request.form.get("urb").lower())
        input_data.append(request.form.get("pf").lower())
        input_data.append(float(request.form.get("nop")))
        input_data.append(float(request.form.get("nof")))
        input_data.append(float(request.form.get("nofs")))

        data = pd.Series(input_data)
        data = data.replace('no', 0)
        data = data.replace('yes', 1)
        inp = np.array(data)
        inp = inp.reshape(1, len(inp))
        
        # Call model function (replace with actual function)
        res = get_result(inp)  # Your ML/DL model logic 
        
        if res:
            result = 'FAKE'
        else:
            result = 'REAL'
        
        return render_template("result.html", result=result)
    return render_template("index.html")

@app.route("/result")
def result():
    return render_template("result.html")

# Running Flask in production mode
import os

if __name__ == "__main__":
    # Use the default port provided by Heroku or Render
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))