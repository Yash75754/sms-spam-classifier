streamlit import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import os

# App title and description
st.title("Email/SMS Spam Classifier")
st.write("This app classifies messages as spam or not spam.")

# Download NLTK resources with a try-except block to handle potential errors
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    st.error(f"Error downloading NLTK resources: {e}")
    st.info("If you're running this locally, make sure you have an internet connection or pre-download the NLTK data.")

ps = PorterStemmer()


# Function to preprocess the text
def transform_text(text):
    # Lowercase the text
    text = text.lower()
    # Tokenize the text
    text = word_tokenize(text)
    # Remove non-alphanumeric characters
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    text = [i for i in text if i not in stop_words and i not in string.punctuation]

    # Stem words
    text = [ps.stem(i) for i in text]

    return " ".join(text)


# Load the trained model and TF-IDF vectorizer with error handling
@st.cache_resource
def load_models():
    try:
        vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        return vectorizer, model, True
    except FileNotFoundError:
        return None, None, False
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, False


tfidf, model, models_loaded = load_models()

# Show warning if models aren't loaded
if not models_loaded:
    st.warning(
        "⚠️ Model files (vectorizer.pkl and model.pkl) not found. Please ensure these files are in the same directory as this script.")
    st.info("These files should contain your trained TF-IDF vectorizer and classification model.")

# Text area to input the message
input_sms = st.text_area("Enter the message to classify:")

# Example button
if st.button('Try an example'):
    input_sms = "Congratulations! You've won a free gift card worth $500. Click here to claim now!"
    st.session_state.input_sms = input_sms
    st.experimental_rerun()

# Predict when the button is pressed
if st.button('Predict'):
    if not input_sms:
        st.warning("Please enter a message.")
    elif not models_loaded:
        st.error("Cannot make predictions without model files.")
    else:
        with st.spinner('Classifying...'):
            # Preprocess the input text
            transformed_sms = transform_text(input_sms)

            # Vectorize the preprocessed text
            vector_input = tfidf.transform([transformed_sms])

            # Predict using the trained model
            result = model.predict(vector_input)[0]

            # Display the result with styling
            if result == 1:
                st.error("📵 SPAM DETECTED")
                st.write("This message appears to be spam.")
            else:
                st.success("✅ NOT SPAM")
                st.write("This message appears to be legitimate.")