import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.utils.helpers import setup_logger

logger = setup_logger("data_preprocessing")

def clean_and_prepare_data(df: pd.DataFrame, config: dict):
    """Cleans, encodes labels, performs feature selection, and splits the data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    config : dict
        Configuration dictionary.
        
    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, df_clean, le, useful_symptoms)
    """
    processed_dir = config["data"]["processed_dir"]
    models_dir = config["data"]["models_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Identify symptom columns
    symptom_cols = [c for c in df.columns if c != 'diseases']
    logger.info(f"Total initial symptom features: {len(symptom_cols)}")
    
    # Ensure they are binary integers (use int8 to save memory)
    df[symptom_cols] = df[symptom_cols].astype(np.int8)
    
    # Quick check for unique values (memory efficient)
    unique_vals = np.unique(df[symptom_cols].values)
    logger.info(f"Unique values in symptom columns: {unique_vals}")
    
    # 2. Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info(f"Dropped {before - len(df)} duplicate rows. Remaining: {len(df)}")
    
    # 3. Drop rows with missing target
    df.dropna(subset=['diseases'], inplace=True)
    
    # Strip whitespace from disease names
    df['diseases'] = df['diseases'].str.strip()

    # Filter out rare classes with only 1 sample (cannot be stratified)
    class_counts = df['diseases'].value_counts()
    rare_classes = class_counts[class_counts < 2].index.tolist()
    if rare_classes:
        logger.warning(f"Filtering out {len(rare_classes)} disease classes with only 1 sample (cannot be stratified): {rare_classes[:10]}...")
        df = df[~df['diseases'].isin(rare_classes)]
        logger.info(f"Remaining records: {len(df)}")
    
    # 4. Target encoding
    le = LabelEncoder()
    df['disease_encoded'] = le.fit_transform(df['diseases'])
    logger.info(f"Total disease classes: {len(le.classes_)}")
    
    # Downsample dataset to 15% of its size (approx 37,000 samples) to prevent MemoryError / ArrayMemoryError on Windows.
    # Keep at least 2 samples per class to ensure stratification works.
    random_state = config["preprocessing"]["random_state"]
    df = df.groupby('disease_encoded', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), max(2, int(len(x) * 0.15))), random_state=random_state)
    )
    logger.info(f"Downsampled dataset to 15% to prevent MemoryError. New shape: {df.shape}")
    
    # Save the label encoder
    le_path = os.path.join(processed_dir, "label_encoder.pkl")
    joblib.dump(le, le_path, compress=3)
    # Also copy to models_dir for inference convenience
    joblib.dump(le, os.path.join(models_dir, "label_encoder.pkl"), compress=3)
    logger.info(f"Label encoder saved to {le_path} and models directory.")
    
    # 5. Feature selection based on variance threshold
    threshold = config["preprocessing"]["symptom_threshold"]
    symptom_freq = df[symptom_cols].mean()
    useful_symptoms = symptom_freq[symptom_freq >= threshold].index.tolist()
    removed = len(symptom_cols) - len(useful_symptoms)
    logger.info(f"Feature selection: removed {removed} low-frequency symptoms (< {threshold*100}% frequency)")
    logger.info(f"Useful symptoms remaining: {len(useful_symptoms)}")
    
    # Save useful symptom list
    feature_cols_path = os.path.join(processed_dir, "feature_columns.csv")
    pd.Series(useful_symptoms).to_csv(feature_cols_path, index=False)
    # Also copy to models_dir
    pd.Series(useful_symptoms).to_csv(os.path.join(models_dir, "feature_columns.csv"), index=False)
    logger.info(f"Feature columns list saved to {feature_cols_path}")
    
    # 6. Define X and y
    X = df[useful_symptoms].values
    y = df['disease_encoded'].values
    
    # 7. Stratified Train / Test split
    test_size = config["preprocessing"]["test_size"]
    random_state = config["preprocessing"]["random_state"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    logger.info(f"Train size: {X_train.shape[0]} rows | Test size: {X_test.shape[0]} rows")
    
    # 8. Save splits
    np.save(os.path.join(processed_dir, "X_train.npy"), X_train)
    np.save(os.path.join(processed_dir, "X_test.npy"), X_test)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test)
    
    # Save clean full processed dataset
    df_clean = df[['diseases', 'disease_encoded'] + useful_symptoms]
    df_clean_path = os.path.join(processed_dir, "cleaned_dataset.csv")
    df_clean.to_csv(df_clean_path, index=False)
    logger.info(f"Preprocessing completed and splits/datasets saved to {processed_dir}")
    
    return X_train, X_test, y_train, y_test, df_clean, le, useful_symptoms
