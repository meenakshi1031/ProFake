from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import re
import joblib
import tensorflow as tf
from ml_dl_fusion import get_result  # Import the fusion logic
import os

# Initialize Flask app
app = Flask(
    __name__,
    template_folder='frontend/templates',
    static_folder='frontend/static'
)

# Load the saved models
knn_model = joblib.load(os.path.join(os.getcwd(), 'knn_model.pkl'))  # Ensure correct path
lstm_model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'lstm_model.h5'))

# Define Flask routes
@app.route("/")
def home():
    return render_template("home.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == 'POST':
        try:
            input_data = []

            # Collect and preprocess user input
            input_data.append(request.form.get("profilepic").lower())
            username = request.form.get("nctulr")
            nctulr = len(re.findall(r'\d+', username)) / len(username) if len(username) > 0 else 0
            input_data.append(nctulr)

            fullname = request.form.get("fnwc")
            fnwc = len(fullname.split())
            input_data.append(float(fnwc))
            nctfnlr = len(re.findall(r'\d+', fullname)) / len(fullname) if len(fullname) > 0 else 0
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

            # Convert input to appropriate format
            data = pd.Series(input_data)
            data = data.replace({'no': 0, 'yes': 1})
            inp = np.array(data).reshape(1, len(data))

            # Get result from model fusion
            prediction = get_result(inp)  # Weighted fusion of KNN and LSTM predictions

            result = 'FAKE' if prediction else 'REAL'
        except Exception as e:
            result = f"Error: {str(e)}"

        return render_template("result.html", result=result)
    return render_template("index.html")

@app.route("/result")
def result_page():
    return render_template("result.html")

# Running Flask app
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


"""from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import re
import joblib
import tensorflow as tf
from ml_dl_fusion import get_result  # Import from the converted Python script

# Initialize Flask app
app = Flask(__name__,template_folder='frontend/templates',static_folder='frontend/static') 

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
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))"""