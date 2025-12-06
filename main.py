import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN
import streamlit as st
max_features = 10000 
# Load model
model = load_model('model.keras')

# Load IMDB word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Decode review (your original logic)
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Preprocess text (YOUR EXACT LOGIC, UNTOUCHED)
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = []

    for word in words:
        # your EXACT behavior: get IMDB index or UNK=2, then add 3
        idx = word_index.get(word, 2) + 3

        # 🔥 required to prevent embedding crash (not a logic change)
        if idx >= max_features:
            idx = 2    # fallback to UNK like IMDB does

        encoded_review.append(idx)

    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


# Prediction function (your logic)
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    return sentiment, prediction[0][0]

# Streamlit UI
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative')

user_input = st.text_area("Movie Review")

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    prediction = model.predict(preprocessed_input)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    st.write(f'Sentiment:  {sentiment}')
    st.write(f'Prediction Score : {prediction[0][0]}')
else:
    st.write("Please Enter a movie Review")
