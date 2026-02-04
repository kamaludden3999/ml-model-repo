# ML Model Repo

This repository contains a minimal example of training and using a scikit-learn model.

Files added:
- `train.py` — training and prediction script that uses the Iris dataset and saves a model with joblib.
- `requirements.txt` — Python dependencies.

Quickstart

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Train the example model (saves to `model.joblib`):

```bash
python train.py --train --model-path model.joblib
```

3. Predict using the saved model (predicts a sample by index):

```bash
python train.py --predict --model-path model.joblib --sample-index 0
```