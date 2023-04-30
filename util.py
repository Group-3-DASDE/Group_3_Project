#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

def load_data():
    # Load the data from the CSV files
    fake_df = pd.read_csv('fake.csv')
    real_df = pd.read_csv('real.csv')

    # Combine the fake and real dataframes
    df = pd.concat([fake_df, real_df])

    # Add a new column to the dataframe called "label"
    df['label'] = np.where(df['title'].str.contains('fake'), 1, 0)

    # Use CountVectorizer to convert the "title" and "text" columns into a matrix of word counts
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['title'] + ' ' + df['text'])

    # Define the target variable (i.e., the "label" column)
    y = df['label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, vectorizer

def build_model(X_train, y_train):
    # Build the decision tree model
    tree = DecisionTreeClassifier(max_depth=3)

    # Fit the model on the training data
    tree.fit(X_train, y_train)

    return tree

def predict(tree, vectorizer, title, text):
    # Convert the input into a matrix of word counts using the same vectorizer used to train the model
    X = vectorizer.transform([title + ' ' + text])

    # Make a prediction using the trained model
    prediction = tree.predict(X)

    # Return the prediction (0 for real, 1 for fake)
    return prediction[0]

