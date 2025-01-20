import streamlit as st
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import xgboost as xgb
import pickle
from .model_handler import ModelHandler
from .utils import preprocess_text

# Set page configuration
st.set_page_config(
    page_title="Mental Health Classifier",
    page_icon="🧠",
    layout="centered"
)

@st.cache_resource
def load_models():
    """Load and cache all required models"""
    handler = ModelHandler()
    return handler

def main():
    st.title("Mental Health Text Classifier")
    st.write("Enter text describing your feelings for classification:")
    
    # Initialize model handler
    model_handler = load_models()
    
    # Text input
    user_input = st.text_area("Your text:", height=150)
    
    # Classification button
    if st.button("Classify"):
        if not user_input.strip():
            st.error("Please enter some text to classify.")
            return
            
        try:
            # Get prediction
            prediction = model_handler.predict(user_input)
            
            # Display results
            st.subheader("Classification Result:")
            st.success(f"Predicted Category: {prediction}")
            
        except Exception as e:
            st.error(f"An error occurred during classification: {str(e)}")
            
if __name__ == "__main__":
    main()
