import argparse
import os
import uvicorn
import numpy as np
import pandas as pd
import joblib

from src.utils.helpers import load_config, setup_logger
from src.data.loader import load_raw_data
from src.data.preprocess import clean_and_prepare_data
from src.data.eda import run_eda
from src.models.train import train_baseline_models, tune_random_forest
from src.models.evaluate import run_evaluation
from src.models.predict import symptom_checker_cli

logger = setup_logger("main_pipeline")

def main():
    parser = argparse.ArgumentParser(description="Symptom-to-Disease Predictor ML Pipeline")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["preprocess", "eda", "train", "cli", "api", "pipeline"],
                        help="Pipeline stage to execute: 'preprocess', 'eda', 'train', 'cli', 'api', or 'pipeline' (runs all ML stages).")
    
    args = parser.parse_args()
    config = load_config()
    
    if args.mode == "preprocess":
        logger.info("Executing Preprocessing Stage...")
        df_raw = load_raw_data(config["data"]["raw_path"])
        clean_and_prepare_data(df_raw, config)
        
    elif args.mode == "eda":
        logger.info("Executing Exploratory Data Analysis Stage...")
        df_raw = load_raw_data(config["data"]["raw_path"])
        run_eda(df_raw, config)
        
    elif args.mode == "train":
        logger.info("Executing Model Training and Evaluation Stage...")
        processed_dir = config["data"]["processed_dir"]
        
        # Check if preprocessing outputs exist
        required_files = ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy", "label_encoder.pkl", "feature_columns.csv", "cleaned_dataset.csv"]
        for f in required_files:
            p = os.path.join(processed_dir, f)
            if not os.path.exists(p):
                raise FileNotFoundError(f"Required preprocessing file '{f}' not found at '{p}'. Please run preprocess stage first.")
                
        # Load splits
        X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
        X_test  = np.load(os.path.join(processed_dir, "X_test.npy"))
        y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
        y_test  = np.load(os.path.join(processed_dir, "y_test.npy"))
        le      = joblib.load(os.path.join(processed_dir, "label_encoder.pkl"))
        feature_names = pd.read_csv(os.path.join(processed_dir, "feature_columns.csv")).iloc[:, 0].tolist()
        df_clean = pd.read_csv(os.path.join(processed_dir, "cleaned_dataset.csv"))
        
        # Train baseline models
        baseline_results = train_baseline_models(X_train, y_train, X_test, y_test, config)
        
        # Tune Random Forest model
        best_rf = tune_random_forest(X_train, y_train, X_test, y_test, config)
        
        # Evaluate model, generate charts, reports, and metadata
        run_evaluation(best_rf, X_train, X_test, y_train, y_test, le, feature_names, df_clean, baseline_results, config)
        
    elif args.mode == "pipeline":
        logger.info("Running complete ML pipeline (Preprocess -> EDA -> Train/Tune/Evaluate)...")
        
        # 1. Preprocess
        df_raw = load_raw_data(config["data"]["raw_path"])
        X_train, X_test, y_train, y_test, df_clean, le, feature_names = clean_and_prepare_data(df_raw, config)
        
        # 2. EDA
        run_eda(df_raw, config)
        
        # 3. Train & Evaluate
        baseline_results = train_baseline_models(X_train, y_train, X_test, y_test, config)
        best_rf = tune_random_forest(X_train, y_train, X_test, y_test, config)
        run_evaluation(best_rf, X_train, X_test, y_train, y_test, le, feature_names, df_clean, baseline_results, config)
        
        logger.info("Pipeline executed successfully!")
        
    elif args.mode == "cli":
        logger.info("Launching Symptom Checker CLI...")
        symptom_checker_cli()
        
    elif args.mode == "api":
        logger.info("Starting FastAPI Server...")
        uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
