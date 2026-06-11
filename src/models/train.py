import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
from src.utils.helpers import setup_logger, batch_predict

logger = setup_logger("model_training")

def train_baseline_models(X_train: np.ndarray, y_train: np.ndarray, 
                          X_test: np.ndarray, y_test: np.ndarray, 
                          config: dict) -> dict:
    """Trains Decision Tree, Random Forest, and XGBoost baselines and plots a comparison.
    
    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training features and targets.
    X_test, y_test : np.ndarray
        Testing features and targets.
    config : dict
        Configuration dictionary.
        
    Returns
    -------
    dict
        Dictionary containing baseline training results.
    """
    models_dir = config["data"]["models_dir"]
    plots_dir = config["data"]["plots_dir"]
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=20, min_samples_leaf=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=30, max_depth=15, min_samples_leaf=4, random_state=42, n_jobs=1),
        "XGBoost"      : XGBClassifier(max_depth=4, n_estimators=10, random_state=42,
                                       use_label_encoder=False, eval_metric='mlogloss',
                                       n_jobs=1, verbosity=0, tree_method='hist')
    }
    
    results = {}
    for name, model in models.items():
        logger.info(f"Training {name} baseline...")
        t0 = time.time()
        try:
            model.fit(X_train, y_train)
            train_time = time.time() - t0
            
            y_pred = batch_predict(model, X_test)
            acc = accuracy_score(y_test, y_pred)
            f1_mac = f1_score(y_test, y_pred, average='macro', zero_division=0)
            f1_wt  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            results[name] = {
                "model": model,
                "y_pred": y_pred,
                "accuracy": acc,
                "f1_macro": f1_mac,
                "f1_weighted": f1_wt,
                "train_time": train_time
            }
            logger.info(f"  {name} Baseline Accuracy: {acc:.4f} | F1-macro: {f1_mac:.4f} | Time: {train_time:.1f}s")
            
            # Save baseline model
            baseline_filename = f"baseline_{name.replace(' ', '_')}.pkl"
            joblib.dump(model, os.path.join(models_dir, baseline_filename), compress=3)
        except (MemoryError, Exception) as e:
            logger.warning(f"  Failed to train {name} baseline due to error: {e}. Skipping baseline entry.")
        
    # Plot baseline comparisons
    names  = list(results.keys())
    accs   = [results[n]["accuracy"] for n in names]
    f1_macs = [results[n]["f1_macro"] for n in names]
    f1_wts  = [results[n]["f1_weighted"] for n in names]
    
    x = np.arange(len(names))
    w = 0.25
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, accs,    w, label="Accuracy",     color="#4C72B0")
    ax.bar(x,     f1_macs, w, label="F1 Macro",     color="#DD8452")
    ax.bar(x + w, f1_wts,  w, label="F1 Weighted",  color="#55A868")
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Baseline Model Comparison")
    ax.legend()
    ax.grid(axis='y', alpha=0.4)
    
    for i, (a, fm, fw) in enumerate(zip(accs, f1_macs, f1_wts)):
        ax.text(i - w, a  + 0.01, f"{a:.3f}",  ha='center', fontsize=8)
        ax.text(i,     fm + 0.01, f"{fm:.3f}", ha='center', fontsize=8)
        ax.text(i + w, fw + 0.01, f"{fw:.3f}", ha='center', fontsize=8)
        
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "model_comparison.png"), dpi=150)
    plt.close()
    
    logger.info(f"Baseline comparison plot saved to: {os.path.join(plots_dir, 'model_comparison.png')}")
    return results

def tune_random_forest(X_train: np.ndarray, y_train: np.ndarray, 
                       X_test: np.ndarray, y_test: np.ndarray, 
                       config: dict) -> RandomForestClassifier:
    """Tunes a Random Forest classifier using RandomizedSearchCV based on config settings.
    
    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training features and targets.
    X_test, y_test : np.ndarray
        Testing features and targets.
    config : dict
        Configuration dictionary.
        
    Returns
    -------
    RandomForestClassifier
        Tuned Random Forest estimator.
    """
    models_dir = config["data"]["models_dir"]
    tuning_config = config["models"]["rf_tuning"]
    
    param_dist = tuning_config["param_grid"]
    n_iter = tuning_config["n_iter"]
    cv = tuning_config["cv"]
    random_state = tuning_config["random_state"]
    
    base_rf = RandomForestClassifier(random_state=random_state, n_jobs=1)
    
    logger.info(f"Starting RandomizedSearchCV for Random Forest (n_iter={n_iter}, cv={cv})...")
    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="f1_weighted",
        random_state=random_state,
        n_jobs=1,
        verbose=1
    )
    
    t0 = time.time()
    search.fit(X_train, y_train)
    tuning_time = time.time() - t0
    logger.info(f"Tuning completed in {tuning_time:.1f}s")
    
    logger.info("Best Parameters found:")
    for k, v in search.best_params_.items():
        logger.info(f"  {k}: {v}")
        
    best_rf = search.best_estimator_
    
    # Evaluate tuned model
    y_pred = batch_predict(best_rf, X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_wt  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    logger.info(f"Tuned RF Accuracy: {acc:.4f} | F1 Weighted: {f1_wt:.4f} | F1 Macro: {f1_mac:.4f}")
    
    # Save the tuned model
    tuned_model_path = os.path.join(models_dir, "best_model_tuned_rf.pkl")
    joblib.dump(best_rf, tuned_model_path, compress=3)
    logger.info(f"Tuned Random Forest saved to: {tuned_model_path}")
    
    return best_rf
