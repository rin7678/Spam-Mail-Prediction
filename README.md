
# Spam SMS Detection

This repository contains the code for a Spam SMS detection system developed during an internship at Afame Technologies. The system uses a machine learning model to classify SMS messages as either spam or ham (non-spam).

## Project Overview

The goal of this project is to build a predictive model to classify SMS messages. The model is trained on a labeled dataset and uses text feature extraction techniques and a logistic regression algorithm to make predictions.

## Files

- `spam_sms.py`: This is the main Python script that contains the implementation of the spam SMS detection system.

## Installation

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/rin7678/Spam-Mail-Prediction.git
    ```

2. Navigate to the project directory:
    ```bash
    cd spam-sms-detection
    ```

3. Install the required dependencies:
    ```bash
    # Importing necessary libraries
*import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score*
    ```

## Usage

1. Ensure you have the dataset `spam.csv` in the same directory as the script.

2. Run the script:
    ```bash
    python spam_sms.py
    ```

3. The script will output the accuracy of the model on the training data and test data. It will also perform a sample prediction on a given input message.

## Dataset

The dataset used for training the model is a CSV file (`spam.csv`) which contains SMS messages labeled as spam or ham. The dataset is preprocessed to remove unnecessary columns and encode the labels.

## Model

The model uses the following steps:

1. **Data Collection & Pre-Processing**: Load the dataset and preprocess it by dropping unnecessary columns and encoding the labels.
2. **Splitting the Data**: Split the data into training and test sets.
3. **Feature Extraction**: Use `TfidfVectorizer` to convert the text data into numerical features.
4. **Training the Model**: Train a logistic regression model on the training data.
5. **Evaluating the Model**: Evaluate the model's accuracy on both the training and test data.
6. **Predictive System**: Build a system to predict whether a given SMS message is spam or ham.

## Example

Here's an example of how the predictive system works:

```python
# Sample input message
input_mail = ["I've been searching for the right words to thank you for this breather. I promise I won't take your help for granted and will fulfill my promise. You have been wonderful and a blessing at all times"]

# Making a prediction
prediction = model.predict(input_data_features)
print(prediction)

# Output
if prediction[0] == 1:
    print('Ham mail')
else:
    print('Spam mail')
```

## Authors ✍️

This project was developed during an internship at Afame Technologies.

