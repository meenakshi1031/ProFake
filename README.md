# ProFake

ProFake is a hybrid model designed to detect fake Instagram accounts using a combination of machine learning and deep learning. 
It combines KNN and LSTM algorithms through weighted fusion to enhance detection accuracy. 
A Flask-based web interface is then deployed.

Model Performance:
Machine Learning (KNN):
- SVM Accuracy: 48.20%
- KNN Accuracy: 89.69%%

Deep Learning (LSTM):
- MLP Accuracy: 76.98%
- LSTM Accuracy: 87.70% (variable due to random weight initialization, data shuffling, and stochastic optimization)

Fusion of KNN and LSTM:
Fusion Accuracy: 91.37% (fluctuates based on individual model performance(lstm) and fusion weights)