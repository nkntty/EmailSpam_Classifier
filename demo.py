import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import streamlit as st

# neural network architectures
class SpamClassifier(nn.Module):
    def __init__(self, input_dim=57, dropout_rate=0.2, hidden_dim=128):
        super(SpamClassifier, self).__init__()
        
        self.layers = nn.Sequential(
            # hidden layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # hidden layer 2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # hidden layer 3
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),
            
            # output layer
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

# Aitua's architecture
class TunedSpamClassifier(nn.Module):
    def __init__(self, input_dim=57, dropout_rate=0.2, hidden_dim=64):
        super(TunedSpamClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),

            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# Esther's architecture uses .net and no sigmoid
class EstherSpamClassifier(nn.Module):
    def __init__(self, input_dim=57):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.35),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.35),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# page
st.set_page_config(page_title="Email Classifier", layout="wide")
st.title("Email Classifier Demo")
st.write("Neural Network Models for Spam Detection")

model_configs = {
    "Danylo's NN": {
        "file": "danylo.pth",
        "architecture": SpamClassifier,
        "has_sigmoid": True
    },
    "Aitua's NN": {
        "file": "aitua.pth",
        "architecture": TunedSpamClassifier,
        "has_sigmoid": True
    },
    "Leshawn's NN": {
        "file": "leshawn.pth",
        "architecture": SpamClassifier,
        "has_sigmoid": True
    },
    "Tatsuya's NN": {
        "file": "tatsuya.pth",
        "architecture": SpamClassifier,
        "has_sigmoid": True
    },
    "Esther's NN": {
        "file": "esther.pth",
        "architecture": EstherSpamClassifier,
        "has_sigmoid": False  # BCEWithLogitsLoss
    }
}

@st.cache_resource

@st.cache_resource
def load_model(model_name):
    config = model_configs[model_name]
    model_path = Path(__file__).parent / "models" / config["file"]

    checkpoint = torch.load(model_path, map_location="cpu")
    model = config["architecture"]()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint.get('test_acc', 'N/A'), config["has_sigmoid"]


# model selection
st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose a model:",
    options=list(model_configs.keys())
)

##st.sidebar.write("Selected model:", selected_model)
st.sidebar.write("Selected model:", selected_model)
st.sidebar.write("Config:", model_configs[selected_model])




# selected model
model, test_acc, has_sigmoid = load_model(selected_model)
st.sidebar.success(f"Model loaded successfully!")
st.sidebar.metric("Test Accuracy", f"{test_acc:.2%}" if isinstance(test_acc, float) else test_acc)

# predict based on sigmoid
def get_preds(model, X_tensor, has_sigmoid):
    with torch.no_grad():
        outputs = model(X_tensor)
        if has_sigmoid:
            # outputs probabilities
            probabilities = outputs.numpy().flatten()
        else:
            # outputs logits
            probabilities = torch.sigmoid(outputs).numpy().flatten()
        predictions = (probabilities > 0.5).astype(int)
    return predictions, probabilities

# input
st.header("Input Email Features")

uploaded_file = st.file_uploader("Upload a CSV file with email features", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data")
    st.dataframe(df.head())
        
    if st.button("Classify Emails"):
        # remove 'class' column
        X = df.drop('class', axis=1) if 'class' in df.columns else df

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        # pred
        predictions, probabilities = get_preds(model, X_tensor, has_sigmoid)
        
        # results
        results_df = df.copy()
        results_df['Prediction'] = ['Spam' if p == 1 else 'Ham' for p in predictions]
        results_df['Confidence'] = [f"{p:.2%}" if predictions[i] == 1 else f"{1-p:.2%}" for i, p in enumerate(probabilities)]
        st.write("### Predictions")
        st.dataframe(results_df)
        spam_count = sum(predictions)
        st.metric("Spam Emails Detected", f"{spam_count} / {len(predictions)}")