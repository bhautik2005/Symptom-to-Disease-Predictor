# 🏥 Symptom-to-Disease Predictor

A professional, simple machine learning repository for predicting diseases from symptom inputs.
The project includes data preprocessing, exploratory analysis, model training, evaluation, an interactive CLI, and a FastAPI inference API.

---

## 📂 Project Structure

```text
symptom-disease-predictor/
├── data/                      # Raw, processed, and external datasets
├── artifacts/                 # Generated outputs and saved models
│   ├── data/                  # Train/test splits and label arrays
│   ├── models/                # Saved model, encoder, and metadata artifacts
│   ├── plots/                 # Generated analysis and evaluation figures
│   └── reports/               # Performance reports and metadata
├── src/                       # Project source code
├── tests/                     # Unit tests
├── config.yaml                # Pipeline configuration and hyperparameters
├── main.py                    # CLI entrypoint for pipeline modes
├── requirements.txt           # Python dependencies
├── setup.py                   # Project installation support
├── README.md                  # Project documentation
└── ui_img/                    # UI screenshot images
```

---

## 📊 Artifact Plots

The pipeline saves plot images in `artifacts/plots/` for analysis and reporting.
The generated plot files include:

* `disease_distribution.png`
* `top20_symptoms.png`
* `symptom_disease_heatmap.png`
* `symptom_cooccurrence.png`
* `signature_symptoms.png`
* `model_comparison.png`
* `confusion_matrix.png`
* `feature_importance_global.png`
* `feature_importance_per_disease.png`

These charts document dataset structure, symptom relationships, model comparison, and feature importance.

---

## 📷 UI Screenshots

UI images are available in the `ui_img/` folder.
These show how the CLI and inference experience appear in practice.

* `ui_img/Screenshot 2026-06-11 102247.png`
* `ui_img/Screenshot 2026-06-11 102546.png`

![UI screenshot 1](ui_img/Screenshot%202026-06-11%20102247.png)

![UI screenshot 2](ui_img/Screenshot%202026-06-11%20102546.png)

---

## 🚀 Quick Start

### 1. Create and activate the virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 3. Run pipeline stages

```powershell
python main.py --mode preprocess
python main.py --mode eda
python main.py --mode train
```

### 4. Run the full pipeline

```powershell
python main.py --mode pipeline
```

### 5. Run the interactive CLI

```powershell
python main.py --mode cli
```

### 6. Run the FastAPI server

```powershell
python main.py --mode api
```

Open the API docs at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧪 Run Tests

```powershell
python -m unittest discover -s tests
```

---

## 💡 Overview

This repository delivers:

* standardized symptom ingestion and preprocessing
* dataset analysis and generated plot artifacts
* model training, evaluation, and saved model artifacts
* CLI and FastAPI inference interfaces

---

## 📬 Contact

* **Bhautik Gondaliya** — `bhautik613@gmail.com`
