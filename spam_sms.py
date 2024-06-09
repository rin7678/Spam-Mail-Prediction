# -*- coding: utf-8 -*-
"""Spam SMS.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1I7GXwlt-CpJjLKR_isSVQfkpsHJtsOgB
"""

# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data Collection & Pre-Processing

# Load the data from a CSV file into a pandas DataFrame
raw_mail_data = pd.read_csv('/content/spam.csv', encoding='latin-1')

raw_mail_data

columns_to_drop = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
df = raw_mail_data.drop(columns=columns_to_drop)

df

new_column_names = {'v1': 'Catergory', 'v2': 'Messages'}
df = df.rename(columns=new_column_names)

df.head()

df.shape

# Label encoding
df.loc[df['Catergory'] == 'spam', 'Catergory',] = 0
df.loc[df['Catergory'] == 'ham', 'Catergory',] = 1

df.head()

# Separating the data as texts and labels
X = df['Messages']
Y = df['Catergory']

# splitting the data into training data and test data.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Feature extraction
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

# x train features
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Convert y train and y test values as integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# training the ml model
model = LogisticRegression()

# training the logistic regression model with the training data
model.fit(X_train_features, Y_train)

# evaluating the logistic model
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

# find the accuracy score value
print('Accuracy on training data : ', accuracy_on_training_data)

# doing the same prediction on test data
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

# Finding the accuracy on test data
print('Accuracy on test data : ', accuracy_on_test_data)

# building a predicitve system
input_mail = ["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times"]

# converting to feature values
input_data_features = feature_extraction.transform(input_mail)

# making predictions
prediction = model.predict(input_data_features)
print(prediction)

# creating a list
if prediction[0]==1:
  print('Ham mail')
else:
  print('Spam mail')