import streamlit as st
import xgboost as xgb
import pickle
from transformers import BertTokenizer, TFBertModel
import numpy as np
import tensorflow as tf

# Load Model and Label Encoder
@st.cache_resource
def load_model():
    # Load XGBoost model
    model = xgb.Booster()
    model.load_model("xgboost_mental_health.json")

    # Load Label Encoder
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    
    return model, label_encoder

# Load BERT Tokenizer and Model
@st.cache_resource
def load_bert():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = TFBertModel.from_pretrained("bert-base-uncased")
    return tokenizer, bert_model

# Generate BERT Embeddings
def generate_embeddings(text, tokenizer, bert_model):
    inputs = tokenizer([text], max_length=128, padding=True, truncation=True, return_tensors="tf")
    outputs = bert_model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # CLS token embeddings
    return embeddings

# Streamlit App
def main():
    st.title("Mental Health Severity Classifier")
    st.write("Enter a sentence about your feelings, and the model will classify the mental health condition.")

    # Input Text
    user_input = st.text_area("Input Sentence", placeholder="Type something here...")

    if st.button("Classify"):
        if not user_input.strip():
            st.error("Please enter a valid sentence.")
        else:
            # Load models and tokenizer
            model, label_encoder = load_model()
            tokenizer, bert_model = load_bert()

            # Generate embeddings and predict
            embeddings = generate_embeddings(user_input, tokenizer, bert_model)
            prediction = model.predict(embeddings)

            # Decode prediction
            if len(prediction.shape) == 1:
                predicted_class = label_encoder.inverse_transform(prediction.astype(int))[0]
            else:
                predicted_class = label_encoder.inverse_transform(np.argmax(prediction, axis=1))[0]

            # Display Result
            st.success(f"Predicted Mental Health Condition: **{predicted_class}**")

if __name__ == "__main__":
    main()
