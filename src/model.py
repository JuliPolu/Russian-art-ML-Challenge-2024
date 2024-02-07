from transformers import AutoImageProcessor, AutoModel
import torch

from tqdm import tqdm
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

from dataset_process import load_image


def init_model(device, architect):
    processor = AutoImageProcessor.from_pretrained(architect)
    model = AutoModel.from_pretrained(architect)
    model = model.to(device)
    return model, processor


def calc_embeddings(model, x_train, processor, device):
    train_embeddings = []
    with torch.no_grad():
        for data in tqdm(x_train):
            inputs = processor(images=load_image(data), return_tensors="pt").to(device) 
            outputs = model(**inputs)
            image_embeddings = outputs.last_hidden_state
            #take average of all 'tokens'
            avg_embeddings = image_embeddings.mean(dim=1)
            train_embeddings.append(avg_embeddings.detach().cpu().numpy())
    train_embeddings_final = np.vstack(train_embeddings)
    return train_embeddings_final


def train_and_save_classifier(X_train, y_train, model_path, scaler_path):
    """
    Train the SVM classifier and save the model.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    svm_classifier = SVC(kernel='linear', gamma='scale')
    svm_classifier.fit(X_train_scaled, y_train)
    joblib.dump(svm_classifier, model_path)
    joblib.dump(scaler, scaler_path)


def load_and_evaluate_classifier(model_path, scaler_path, X_test, y_test):
    """
    Load the SVM classifier and evaluate it on the test data.
    """
    # Load the model and scaler
    svm_classifier = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    X_test_scaled = scaler.transform(X_test)
    svm_predictions = svm_classifier.predict(X_test_scaled)

    # Calculating F1 Score
    f1_svm = f1_score(y_test, svm_predictions, average='macro')

    return pd.DataFrame({
        'SVM': {'F1 Score': f1_svm},
    })
