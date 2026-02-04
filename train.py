#!/usr/bin/env python3
"""Simple training and prediction script using scikit-learn and the Iris dataset.

Usage:
  - Train: python train.py --train --model-path model.joblib
  - Predict: python train.py --predict --model-path model.joblib --sample-index 0
"""
import argparse
import joblib
import os
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_and_save(model_path: str, test_size: float = 0.2, random_state: int = 42):
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=200)
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")

    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Saved model to {model_path}")

def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

def predict_sample(model_path: str, sample_index: int = 0):
    model = load_model(model_path)
    data = load_iris()
    X, y = data.data, data.target
    if sample_index < 0 or sample_index >= len(X):
        raise IndexError("sample-index out of range")
    sample = X[sample_index:sample_index+1]
    pred = model.predict(sample)[0]
    print(f"Predicted class: {pred}, true class: {y[sample_index]}")

def main():
    parser = argparse.ArgumentParser(description="Train or predict an example model on the Iris dataset")
    parser.add_argument('--train', action='store_true', help='Train a model and save it')
    parser.add_argument('--predict', action='store_true', help='Load a saved model and predict a sample')
    parser.add_argument('--model-path', type=str, default='model.joblib', help='Path to save or load the model')
    parser.add_argument('--sample-index', type=int, default=0, help='Index of sample to predict when using --predict')
    args = parser.parse_args()

    if args.train:
        train_and_save(args.model_path)
    elif args.predict:
        predict_sample(args.model_path, args.sample_index)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()