# Step-by-Step Running Guide

This document contains step-by-step instructions and commands to run the **Symptom-to-Disease Predictor** project.

---

## Step 1: Environment Setup

First, navigate to the project directory in your terminal:
```bash
cd symptom-disease-predictor
```

### 1. Create a Virtual Environment
Choose the command depending on your operating system:

* **Windows**:
  ```bash
  python -m venv venv
  ```
* **macOS / Linux**:
  ```bash
  python3 -m venv venv
  ```

### 2. Activate the Virtual Environment
Activate the environment to ensure packages are installed locally:

* **Windows (Command Prompt / cmd.exe)**:
  ```cmd
  venv\Scripts\activate
  ```
* **Windows (PowerShell)**:
  ```powershell
  venv\Scripts\Activate.ps1
  ```
  *(Note: If you get a permission/execution policy error in PowerShell, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first.)*
* **macOS / Linux**:
  ```bash
  source venv/bin/activate
  ```

### 3. Upgrade Pip & Install Dependencies
Install all required libraries and make the project package editable:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## Step 2: Running the ML Pipeline

The project supports running individual stages or the entire pipeline using the `main.py` entrypoint.

### Option A: Run the Pipeline Step-by-Step (Recommended)

1. **Step 2.1: Preprocessing**
   Cleans the raw dataset, performs feature extraction, encodes the labels, splits the dataset into train/test sets, and saves them in `artifacts/data/`.
   ```bash
   python main.py --mode preprocess
   ```

2. **Step 2.2: Exploratory Data Analysis (EDA)**
   Runs statistical analyses on raw data and generates correlation/cooccurrence plots in `artifacts/plots/`.
   ```bash
   python main.py --mode eda
   ```

3. **Step 2.3: Model Training & Evaluation**
   Trains baseline models (Decision Tree, Naive Bayes, etc.), tunes the Random Forest model using Grid/Random Search, and runs comprehensive evaluations (producing classification reports, confusion matrix plots, and feature importances) in `artifacts/reports/` and `artifacts/plots/`.
   ```bash
   python main.py --mode train
   ```

### Option B: Run the Complete Pipeline in One Command
If you want to run preprocessing, EDA, and model training/tuning/evaluation sequentially automatically:
```bash
python main.py --mode pipeline
```

---

## Step 3: Run the Symptom Checker CLI

After training the model, you can run the interactive CLI in your terminal to enter symptoms and get predictions.

* **Direct execution**:
  ```bash
  python main.py --mode cli
  ```
* **Alternate command** (If you ran `pip install -e .`):
  ```bash
  symptom-predictor --mode cli
  ```

---

## Step 4: Run the FastAPI Web Server

You can launch a FastAPI web server to expose endpoints for remote symptom predictions.

```bash
python main.py --mode api
```

Once the server is running:
* Access the web API at `http://127.0.0.1:8000`
* Open the interactive OpenAPI documentation to test endpoints directly from the browser:
  [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Step 5: Run Unit Tests

To run the project test suite and verify everything is working correctly, run the standard library unittest discover command:
```bash
python -m unittest discover -s tests
```
