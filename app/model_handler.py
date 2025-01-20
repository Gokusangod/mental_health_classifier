import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import xgboost as xgb
import pickle
import os
import streamlit as st

class ModelHandler:
    def __init__(self):
        # Load BERT components
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-uncased')
        
        # Load XGBoost model
        model_path = os.path.join('models', 'xgboost_mental_health.json')
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(model_path)
        
        # Load label encoder
        with open(os.path.join('models', 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
    
    def get_bert_embeddings(self, text):
        """Generate BERT embeddings for input text"""
        inputs = self.tokenizer(text, return_tensors="tf", 
                              truncation=True, max_length=512,
                              padding="max_length")
        outputs = self.bert_model(inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()
    
    def predict(self, text):
        """Generate prediction for input text"""
        # Get BERT embeddings
        embeddings = self.get_bert_embeddings(text)
        
        # Convert to DMatrix for XGBoost
        dmatrix = xgb.DMatrix(embeddings)
        
        # Get prediction
        pred_probs = self.xgb_model.predict(dmatrix)
        pred_class = pred_probs.argmax(axis=1)[0]
        
        # Convert to label
        return self.label_encoder.inverse_transform([pred_class])[0]