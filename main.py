#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from util import load_data, build_model, predict

# Load the data and train the model
X_train, X_test, y_train, y_test, vectorizer = load_data()
tree = build_model(X_train, y_train)

# Set up the Streamlit app
st.title('Fake News Classifier')

title = st.text_input('Enter the title of the news article:')
text = st.text_area('Enter the text of the news article:', height=300)

if st.button('Predict'):
    prediction = predict(tree, vectorizer, title, text)

    if prediction == 0:
        st.write('This news article is *real*!')
    else:
        st.write('This news article is *fake*!')

