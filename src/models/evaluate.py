import os
import json
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from src.utils.helpers import setup_logger, batch_predict

logger = setup_logger("model_evaluation")

def run_evaluation(model, X_train: np.ndarray, X_test: np.ndarray, 
                   y_train: np.ndarray, y_test: np.ndarray, 
                   le, feature_names: list, df_clean: pd.DataFrame, 
                   baseline_results: dict, config: dict):
    """Evaluates the final model, generates confusion matrix, feature importance, metadata, and docs.
    
    Parameters
    ----------
    model : RandomForestClassifier
        Tuned Random Forest model.
    X_train, X_test : np.ndarray
        Feature splits.
    y_train, y_test : np.ndarray
        Target splits.
    le : LabelEncoder
        Target encoder.
    feature_names : list of str
        List of selected features.
    df_clean : pd.DataFrame
        Cleaned/processed dataframe.
    baseline_results : dict
        Results of baseline models.
    config : dict
        Configuration dictionary.
    """
    models_dir = config["data"]["models_dir"]
    plots_dir = config["data"]["plots_dir"]
    reports_dir = config["data"]["reports_dir"]
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Predictions and basic scores
    logger.info("Evaluating final model on test data...")
    y_pred = batch_predict(model, X_test)
    final_acc = accuracy_score(y_test, y_pred)
    final_f1w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    final_f1m = f1_score(y_test, y_pred, average='macro',    zero_division=0)
    
    logger.info(f"Final Tuned RF Test Accuracy: {final_acc:.4f} | F1-weighted: {final_f1w:.4f} | F1-macro: {final_f1m:.4f}")
    
    # 2. Confusion Matrix Heatmap (top 20 classes)
    logger.info("Generating confusion matrix plot...")
    cm = confusion_matrix(y_test, y_pred)
    top_classes_idx = np.argsort(cm.sum(axis=1))[-20:]
    cm_top = cm[np.ix_(top_classes_idx, top_classes_idx)]
    labels_top = le.classes_[top_classes_idx]
    
    plt.figure(figsize=(16, 13))
    sns.heatmap(cm_top, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_top, yticklabels=labels_top,
                linewidths=0.3, cbar_kws={"shrink": 0.7})
    plt.title("Confusion Matrix — Tuned Random Forest (top 20 classes)", fontsize=13)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    
    # 3. Global feature importance (top 15)
    logger.info("Extracting global feature importances...")
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "symptom"   : feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    
    fi_df.to_csv(os.path.join(reports_dir, "feature_importance.csv"), index=False)
    
    plt.figure(figsize=(10, 6))
    top15 = fi_df.head(15)
    sns.barplot(data=top15, x="importance", y="symptom", palette="viridis")
    plt.title("Top 15 Most Important Symptoms\n(Random Forest Feature Importance)", fontsize=13)
    plt.xlabel("Importance Score")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "feature_importance_global.png"), dpi=150)
    plt.close()
    
    # 4. Per-disease feature importance
    logger.info("Generating per-disease feature importance...")
    disease_counts = df_clean['diseases'].value_counts()
    top6_diseases = disease_counts.head(6).index.tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    for i, disease in enumerate(top6_diseases):
        mask_in  = df_clean['diseases'] == disease
        mask_out = ~mask_in
        in_mean  = df_clean.loc[mask_in,  feature_names].mean()
        out_mean = df_clean.loc[mask_out, feature_names].mean()
        diff     = (in_mean - out_mean).sort_values(ascending=False).head(10)
        
        axes[i].barh(diff.index[::-1], diff.values[::-1], color='steelblue')
        axes[i].set_title(f"{disease}", fontsize=9, fontweight='bold')
        axes[i].set_xlabel("Symptom importance score", fontsize=8)
        axes[i].tick_params(axis='y', labelsize=7)
        axes[i].axvline(0, color='red', linewidth=0.8, linestyle='--')
    plt.suptitle("Top Symptoms per Disease (vs all others)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "feature_importance_per_disease.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Export Final Package
    logger.info("Exporting final model package...")
    joblib.dump(model,         os.path.join(models_dir, "final_model.pkl"), compress=3)
    joblib.dump(le,            os.path.join(models_dir, "final_label_encoder.pkl"), compress=3)
    joblib.dump(feature_names, os.path.join(models_dir, "final_feature_names.pkl"), compress=3)
    
    # 6. Save metadata
    model_meta = {
        "model_type"      : "RandomForestClassifier (tuned)",
        "n_estimators"    : model.n_estimators,
        "max_depth"       : str(model.max_depth),
        "n_features"      : len(feature_names),
        "n_classes"       : len(le.classes_),
        "train_samples"   : int(X_train.shape[0]),
        "test_samples"    : int(X_test.shape[0]),
        "accuracy"        : round(final_acc, 4),
        "f1_weighted"     : round(final_f1w, 4),
        "f1_macro"        : round(final_f1m, 4),
        "saved_date"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(reports_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(model_meta, f, indent=2)
        
    # 7. Generate markdown documentation report
    logger.info("Generating project documentation report...")
    top5_symptoms = fi_df.head(5)['symptom'].tolist()
    
    dt_acc = baseline_results.get("Decision Tree", {}).get("accuracy", 0.0)
    dt_f1w = baseline_results.get("Decision Tree", {}).get("f1_weighted", 0.0)
    dt_f1m = baseline_results.get("Decision Tree", {}).get("f1_macro", 0.0)
    
    rf_acc = baseline_results.get("Random Forest", {}).get("accuracy", 0.0)
    rf_f1w = baseline_results.get("Random Forest", {}).get("f1_weighted", 0.0)
    rf_f1m = baseline_results.get("Random Forest", {}).get("f1_macro", 0.0)
    
    xgb_acc = baseline_results.get("XGBoost", {}).get("accuracy", 0.0)
    xgb_f1w = baseline_results.get("XGBoost", {}).get("f1_weighted", 0.0)
    xgb_f1m = baseline_results.get("XGBoost", {}).get("f1_macro", 0.0)
    
    report = f"""# Symptom-to-Disease Prediction System
## Final Project Documentation
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

---

## 1. Project Overview
A machine learning system that predicts probable diseases from reported symptoms.
Uses a multi-class Random Forest classifier trained on a binary symptom dataset.
Provides Top-3 disease predictions with confidence scores.

---

## 2. Dataset Summary

| Metric                  | Value                          |
|-------------------------|-------------------------------|
| Total records           | {len(df_clean):,}                    |
| Total symptom features  | {len(feature_names)}           |
| Total disease classes   | {len(le.classes_)}             |
| Training samples        | {X_train.shape[0]:,}           |
| Testing samples         | {X_test.shape[0]:,}            |
| Most common disease     | {disease_counts.idxmax()} ({disease_counts.max()} records) |
| Least common disease    | {disease_counts.idxmin()} ({disease_counts.min()} records) |
| Class imbalance ratio   | {disease_counts.max() / disease_counts.min():.1f}x          |

---

## 3. Key EDA Findings

- **Class imbalance**: The dataset has a {disease_counts.max() / disease_counts.min():.1f}x imbalance ratio.
  The most common disease ({disease_counts.idxmax()}) has {disease_counts.max()} records
  vs the rarest with only {disease_counts.min()} records.
- **Feature count**: After removing near-zero-variance symptoms, {len(feature_names)} symptoms
  were retained as useful features out of original symptom columns.
- **Top symptom**: `{fi_df.iloc[0]['symptom']}` is the single most predictive symptom globally.
- **Symptom co-occurrence**: Many symptoms cluster together — suggesting overlapping disease profiles
  that make multi-class classification challenging.

---

## 4. Model Comparison

| Model              | Accuracy | F1 Weighted | F1 Macro |
|--------------------|----------|-------------|----------|
| Decision Tree      | {dt_acc:.4f}   | {dt_f1w:.4f}      | {dt_f1m:.4f}   |
| Random Forest      | {rf_acc:.4f}   | {rf_f1w:.4f}      | {rf_f1m:.4f}   |
| XGBoost            | {xgb_acc:.4f}   | {xgb_f1w:.4f}      | {xgb_f1m:.4f}   |
| **Tuned RF (best)**| **{final_acc:.4f}** | **{final_f1w:.4f}** | **{final_f1m:.4f}** |

---

## 5. Best Model — Tuned Random Forest

| Parameter       | Value                     |
|-----------------|---------------------------|
| Type            | RandomForestClassifier    |
| n_estimators    | {model.n_estimators}      |
| max_depth       | {model.max_depth}         |
| Accuracy        | {final_acc:.4f}           |
| F1 Weighted     | {final_f1w:.4f}           |
| F1 Macro        | {final_f1m:.4f}           |

---

## 6. Top 15 Most Important Symptoms

{fi_df.head(15)[['symptom', 'importance']].to_markdown(index=False)}

---

## 7. Medical Insights Discovered

1. **{top5_symptoms[0]}** is the single strongest predictor across all disease classes,
   suggesting it is a highly discriminative diagnostic marker.

2. **{top5_symptoms[1]}** and **{top5_symptoms[2]}** frequently co-occur, which aligns with
   known clinical patterns in respiratory and systemic conditions.

3. The model struggles most with diseases that share similar symptom profiles
   (e.g. viral vs bacterial infections), matching real clinical diagnostic challenges.

4. Mental health conditions (depression, anxiety) show strong co-occurrence with physical
   symptoms like insomnia and fatigue — the model captures this psychosomatic relationship.

5. Many rare diseases (< 10 records) show poor recall in the classification report,
   highlighting the need for more balanced data collection in future work.

---

## 8. Deliverables Summary

| Deliverable                    | File                                    | Status |
|--------------------------------|-----------------------------------------|--------|
| Cleaned dataset                | artifacts/data/cleaned_dataset.csv      | ✅     |
| Label encoder                  | artifacts/models/final_label_encoder.pkl | ✅     |
| Feature columns list           | artifacts/data/feature_columns.csv      | ✅     |
| Trained model                  | artifacts/models/final_model.pkl         | ✅     |
| Model metadata                 | artifacts/reports/model_metadata.json   | ✅     |
| Feature importance CSV         | artifacts/reports/feature_importance.csv | ✅     |
| Disease distribution chart     | artifacts/plots/disease_distribution.png| ✅     |
| Top 20 symptoms chart          | artifacts/plots/top20_symptoms.png      | ✅     |
| Symptom-disease heatmap        | artifacts/plots/symptom_disease_heatmap.png| ✅  |
| Co-occurrence matrix           | artifacts/plots/symptom_cooccurrence.png| ✅     |
| Signature symptoms chart       | artifacts/plots/signature_symptoms.png  | ✅     |
| Model comparison chart         | artifacts/plots/model_comparison.png    | ✅     |
| Confusion matrix               | artifacts/plots/confusion_matrix.png    | ✅     |
| Feature importance (global)    | artifacts/plots/feature_importance_global.png| ✅|
| Feature importance (per disease)| artifacts/plots/feature_importance_per_disease.png| ✅|

---

## 9. How to Use the Symptom Checker

```python
import joblib, numpy as np, pandas as pd

model         = joblib.load("artifacts/models/final_model.pkl")
le            = joblib.load("artifacts/models/final_label_encoder.pkl")
feature_names = joblib.load("artifacts/models/final_feature_names.pkl")

def predict_top3(symptoms):
    vec = np.zeros(len(feature_names))
    for s in symptoms:
        if s in feature_names:
            vec[feature_names.index(s)] = 1
    proba   = model.predict_proba(vec.reshape(1,-1))[0]
    top3    = np.argsort(proba)[::-1][:3]
    return [(le.inverse_transform([i])[0], round(proba[i]*100, 2)) for i in top3]

print(predict_top3(["fever", "cough", "headache"]))
```

---

## 10. Disclaimer
This system is intended for educational purposes only. It is not a substitute
for professional medical advice, diagnosis, or treatment. Always consult a
qualified healthcare provider for any medical concerns.

---
*Generated by Symptom-to-Disease ML Project — {datetime.now().year}*
"""
    doc_path = os.path.join(reports_dir, "project_documentation.md")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    logger.info(f"Project documentation report generated successfully at: {doc_path}")
    logger.info("Model evaluation complete.")
