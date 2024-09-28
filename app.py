import pandas as pd
import streamlit as st
from PIL import Image
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import PorterStemmer, WordNetLemmatizer
import nltk
import pickle

# Download necessary nltk data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# Load the pre-trained model (assuming it's a pickled file)
def load_model():
    with open('model/model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Define a function for prediction using the loaded model
def prediction(tweet):
    model = load_model()
    # Preprocess the tweet (you can add your custom preprocessing here)
    tweet_cleaned = preprocess_text(tweet)
    # Vectorize the input (you can adjust this based on your TF-IDF training)
    vectorizer = TfidfVectorizer(max_features=1000)
    tweet_vectorized = vectorizer.transform([tweet_cleaned])
    prediction = model.predict(tweet_vectorized)
    return prediction[0]

# Preprocess tweet function (add your own logic here)
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#', '', text)  # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Streamlit app starts here

# Page title and sidebar information
st.sidebar.title("DATA SCIENCE PROJECT")
st.sidebar.write("Data Science Students")
st.sidebar.write("You can reach me at:")
st.sidebar.subheader("pnandhiniofficial@gmail.com & hirthicksofficial@gmail.com")

# Load and display profile image
image_path = 'me.jpg'  # Make sure this image is placed in the same directory or adjust the path
try:
    image = Image.open(image_path)
    st.sidebar.image(image, use_column_width=True)
except:
    st.sidebar.write("Profile image not found.")

# App main title and description
st.write('''
# Cyberbullying Tweet Recognition App

This app predicts the nature of the tweet into 6 Categories.
* Age
* Ethnicity
* Gender
* Religion
* Other Cyberbullying
* Not Cyberbullying

***
''')

# Load and display the main image
try:
    image = Image.open('statics/twitter.png')  # Adjust the image path based on your structure
    st.image(image, use_column_width=True)
except:
    st.write("Main image not found.")

# Text Box for tweet input
st.header('Enter Tweet ')
tweet_input = st.text_area("Tweet Input", height=150)

# Prediction logic
if tweet_input:
    st.header("Predicting...")

    # Call the prediction function
    pred = prediction(tweet_input)

    # Display results
    if pred == "age":
        st.image("statics/Age.png", use_column_width=True)
    elif pred == "ethnicity":
        st.image("statics/Ethnicity.png", use_column_width=True)
    elif pred == "gender":
        st.image("statics/Gender.png", use_column_width=True)
    elif pred == "other_cyberbullying":
        st.image("statics/Other.png", use_column_width=True)
    elif pred == "religion":
        st.image("statics/Religion.png", use_column_width=True)
    elif pred == "not_cyberbullying":
        st.image("statics/not_cyber.png", use_column_width=True)
else:
    st.write('''***No Tweet Text Entered!***''')

st.write('''***''')
