# ProFake

This website uses machine learning and deep learning models to detect Fake Instagram accounts based on user profile data. It integrates a K-Nearest Neighbors (KNN) model and a Long Short-Term Memory (LSTM) model to classify accounts as "REAL" or "FAKE". The models are combined using a weighted fusion approach to improve prediction accuracy.

## Technologies Used

- **Flask**: For the backend web framework.
- **TensorFlow**: For loading and running the LSTM deep learning model.
- **scikit-learn**: For loading and running the KNN machine learning model.
- **NumPy & Pandas**: For data processing and manipulation.
- **HTML/CSS**: For the frontend user interface.
- **Joblib**: For saving and loading machine learning models.
- **TensorFlow Keras**: For working with the deep learning model.

## How It Works

- The user provides details about an Instagram account.
- The get_result function processes the input data and passes it through the KNN and LSTM models.
- Both models generate probabilities indicating whether the account is fake or real.
- The predictions from both models are combined using a weighted fusion approach.
- The final classification (fake or real) is determined and displayed to the user.

# Model Accuracy

Machine Learning (KNN):
- KNN Accuracy: 89.69%

Deep Learning (LSTM):
- LSTM Accuracy: 87-90% (may fluctuate slightly due to the inherent randomness in the model's training process and variations in the input data.)

Fusion of KNN and LSTM: Weighted Fusion
- 88-91% (may fluctuate slightly due to variations in the LSTM modelodel)

## Models

### KNN Model (`knn_model.pkl`)
- A K-Nearest Neighbors (KNN) classifier is trained on labeled Instagram profile data to classify accounts based on various features like username, full name, etc.

### LSTM Model (`lstm_model.h5`)
- An LSTM model is used to analyze sequential data from Instagram profiles, enhancing the prediction for more complex patterns.

### Fusion of Models
- A weighted fusion strategy is used to combine the predictions of both models (KNN and LSTM) for better overall performance.