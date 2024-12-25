
# coding: utf-8

# **Introduction**

# Determining the legitimacy of an Instagram account is challenging. However, a predictive model can be constructed to gauge the probability of an account being fake, based on specific attributes and activity patterns. The objective is to classify user accounts as either genuine or counterfeit, constituting a binary classification task.
# 
# In the digital era, Instagram has emerged as a prominent platform for social interaction, personal expression, and brand establishment. Yet, alongside its popularity, the platform has witnessed a surge in fake profiles and fraudulent activities. These accounts serve various nefarious purposes, including dissemination of misinformation, phishing scams, and identity theft. Addressing this concern necessitates the application of machine learning techniques to automatically detect and eliminate fake profiles from the platform.
# 
# 

# **Goal**

# The objective is to determine if a user account is genuine or fake, which is classified as a binary classification problem due to the presence of two categories. Here, employ a combination of machine learning and deep learning models to leverage the strengths of both approaches.

# # Import Libraries

# In[53]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import  accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from keras import utils
from keras import layers


# # Load data

# The initial step involves loading the data and comprehending the dataset's information structure. Following that, we implement any required alterations to the dataset before proceeding with exploratory data analysis and modeling. There are two files available: one for training and another for testing. The target vector is determined by the 'fake' column.

# In[54]:



train = pd.read_csv('kaggle_dataset/train.csv') # load Training dataset
test = pd.read_csv('kaggle_dataset/test.csv') # load Testing dataset
#combine data
df = pd.concat([train, test], ignore_index=True)


# the features are:
# - Profile Picture: Binary indicator of whether an account has a profile picture or not.
# 
# - Numerical Characters to Username Length Ratio: Continuous feature representing the proportion of numerical characters in an account's username.
# 
# - Full Name Word Count: Continuous feature indicating the total number of words in the person's full name.
# 
# - Numerical Characters to Full Name Length Ratio: Continuous feature indicating the ratio of numerical characters to the total length of the person's full name.
# 
# - Name Matches Username: Binary feature indicating whether the person's name matches their username.
# 
# - Description Length: Length of the profile description, likely the bio.
# 
# - External URL in Bio: Binary feature indicating whether a profile has a link to an external website in its bio.
# 
# - Private Profile: Binary feature indicating whether the profile is restricted to non-followers.
# 
# - Number of Posts: Continuous feature representing the total number of posts on the profile.
# 
# - Number of Followers: Continuous feature representing the total number of followers for each account.
# 
# - Number of Follows: Continuous feature representing the total number of accounts that the user is following.
# 
# - Fake Account: Target variable indicating whether an account is fake or not.

# In[55]:


df.head(4)


# #Exploratory Data Analysis

# In[56]:


# dataset size
print(" size of dataset :",len(df))


# In[57]:


# target count
target_count = df.fake.value_counts()
target_count


# In[58]:


print('target have {}% for non-fake and {}% for fake.'.format(round(100*(target_count[1]/target_count.sum())),
                                                                  round(100*(target_count[0]/target_count.sum()))))


# In[59]:


df.shape


# In[60]:


# graphical analysis
plt.figure(dpi=60,figsize=(12,4))
sns.countplot(x ='fake', data=df, hue = "fake")
plt.xticks(rotation =80)
plt.title('dataset label count')
plt.show()


# In[61]:


# correlation between features
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

plt.show()


# In[62]:


# Check descriptive statistics
df.describe()


# In[63]:


df.info()


# #Preprocessing

# In[64]:


# Check  the missing values

percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'percent_missing (%)': percent_missing})
missing_value_df.sort_values('percent_missing (%)', ascending=False)


# In[65]:


# check nan values
df.isna().any()


# In[66]:


# check  duplicated rows
df_duplicated = df[df.duplicated(keep=False)]
df_duplicated


# In[67]:


# Removing duplicates
df.drop_duplicates(inplace = True)


# In[68]:


# new dataset size
print(" size of dataset :",len(df))


# In[69]:


target=np.array(df['fake'])
data = df.drop('fake',axis=1)
inp_data = np.array(data)
inp_data = np.float64(inp_data)


# # dataspliting

# In[70]:


# for machine learning
xtrain,xtest,ytrain,ytest = train_test_split(inp_data,target,test_size=0.2,random_state=100)


# In[71]:


# for deep learning
x_train = xtrain.reshape((xtrain.shape[0],1,xtrain.shape[1]))
x_test = xtest.reshape((xtest.shape[0],1,xtest.shape[1]))


# #SVM classifier (Machine Learning)

# In[72]:




# svm training
svm = SVC()
svm.fit(xtrain,ytrain)


# In[73]:


predictions = svm.predict(xtest)
svm_acc = accuracy_score(ytest, predictions)
print("Accuracy svm : ",svm_acc)


# In[74]:


cm = confusion_matrix(ytest,predictions)
print("Confusion Matrix:\n{}".format(cm))
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d")
plt.show()


# #KNN classifier (Machine Learning)

# In[75]:


knn_mlmodel = KNeighborsClassifier(2).fit(xtrain, ytrain)
predictions = knn_mlmodel.predict(xtest)
acc = accuracy_score(ytest,predictions)
print("Accuracy knn : ", acc)


# In[76]:



cm = confusion_matrix(ytest,predictions)
print("Confusion Matrix:\n{}".format(cm))
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d")
plt.show()


# #Multi-Layer Perceptron Classifier

# In[77]:



clf = MLPClassifier(random_state=1, max_iter=300).fit(xtrain, ytrain)


# In[78]:


predictions = clf.predict(xtest)
clf_acc = accuracy_score(ytest,predictions)
print("Accuracy MLP Classifier : ", clf_acc)


# In[79]:


cm = confusion_matrix(ytest,predictions)
print("Confusion Matrix:\n{}".format(cm))
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d")
plt.show()


# #LSTM

# In[80]:


lstm_dlmodel = keras.Sequential()
lstm_dlmodel.add(layers.Conv1D(128,1, activation='relu',input_shape=(x_train.shape[1],11)))
lstm_dlmodel.add(layers.LSTM(25,return_sequences=True))
lstm_dlmodel.add(layers.SimpleRNN(20))
lstm_dlmodel.add(layers.Dense(400))
lstm_dlmodel.add(layers.ELU())
lstm_dlmodel.add(layers.Dropout(0.2))
lstm_dlmodel.add(layers.Dense(2, activation='softmax'))
lstm_dlmodel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(lstm_dlmodel.summary())


# In[81]:


history = lstm_dlmodel.fit(x_train,ytrain,epochs=100, validation_split=0.2)


# In[82]:


plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[83]:


pred = lstm_dlmodel.predict(x_test)
pred = np.argmax(pred,axis=1)
acc = accuracy_score(ytest, pred)
cm = confusion_matrix(ytest, pred)
print("        Accuracy: {:.2f}%".format(acc*100))
print("Confusion Matrix:\n{}".format(cm))
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d",cmap='RdYlGn',linewidths=0.5)
plt.show()


# In[84]:


import joblib
joblib.dump(knn_mlmodel, 'knn_model.pkl')  
lstm_dlmodel.save('lstm_model.h5') 


# #fusion

# In[85]:


# Import the necessary libraries
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models 
import tensorflow

# Load the models
knn_model = joblib.load('knn_model.pkl')  # Load KNN model
lstm_model = tf.keras.models.load_model('lstm_model.h5')  # Load LSTM model

# Define get_result function
def get_result(xtest, knn_model, lstm_model):
    pb2 = knn_model.predict_proba(xtest)  # Get KNN probabilities
    x_test = xtest.reshape((xtest.shape[0], 1, xtest.shape[1]))  # Reshape for LSTM
    pb1 = lstm_model.predict(x_test)  # Get LSTM predictions
    w1, w2 = 0.7, 0.2  # Weights for combining the models
    pb = w1 * pb1 + w2 * pb2  # Combine the predictions

    pred = np.argmax(pb, axis=1)  # Get the final prediction
    return pred
  
pred = get_result(xtest)
acc = accuracy_score(ytest, pred)
cm = confusion_matrix(ytest, pred)
print("        Accuracy: {:.2f}%".format(acc*100))
print("Confusion Matrix:\n{}".format(cm))
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d",cmap='RdYlGn',linewidths=0.5)
plt.show()  



# In[86]:


xtest.shape


# In[88]:


#get_ipython().system('jupyter nbconvert --to script ml_dl_fusion.ipynb')

