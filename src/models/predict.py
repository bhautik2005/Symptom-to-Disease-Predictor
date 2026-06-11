import os
import joblib
import numpy as np
import pandas as pd
from src.utils.helpers import setup_logger, load_config

logger = setup_logger("predictor")

class SymptomPredictor:
    def __init__(self, models_dir: str = None):
        """Initializes the symptom predictor by loading models and feature names.
        
        Parameters
        ----------
        models_dir : str, optional
            Path to the directory containing model files. If None, loaded from config.
        """
        if models_dir is None:
            config = load_config()
            models_dir = config["data"]["models_dir"]
            
        model_path = os.path.join(models_dir, "final_model.pkl")
        le_path = os.path.join(models_dir, "final_label_encoder.pkl")
        features_path = os.path.join(models_dir, "final_feature_names.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please run training first.")
            
        self.model = joblib.load(model_path)
        self.le = joblib.load(le_path)
        
        # Load feature columns (saved as list in pickle file)
        if features_path.endswith(".pkl"):
            self.feature_names = joblib.load(features_path)
        else:
            self.feature_names = pd.read_csv(features_path).iloc[:, 0].tolist()
            
        logger.info(f"SymptomPredictor loaded successfully with {len(self.feature_names)} features and {len(self.le.classes_)} target classes.")

    def predict_top3(self, symptom_list: list) -> list:
        """Predicts the top-3 probable diseases for the given symptom list.
        
        Parameters
        ----------
        symptom_list : list of str
            List of symptom strings.
            
        Returns
        -------
        list of dict
            Sorted list of dicts with keys: rank, disease, confidence.
        """
        input_vector = np.zeros(len(self.feature_names), dtype=int)
        unrecognised = []

        for symptom in symptom_list:
            symptom_clean = symptom.strip().lower()
            if symptom_clean in self.feature_names:
                idx = self.feature_names.index(symptom_clean)
                input_vector[idx] = 1
            else:
                unrecognised.append(symptom)

        if unrecognised:
            logger.warning(f"Unrecognised symptoms (ignored): {unrecognised}")

        if input_vector.sum() == 0:
            return [{"rank": 1, "disease": "No valid symptoms entered", "confidence": 0.0}]

        # Get probability scores for all classes
        proba = self.model.predict_proba(input_vector.reshape(1, -1))[0]

        # Get indices of top-3 probabilities
        top3_idx = np.argsort(proba)[::-1][:3]

        results = []
        for rank, idx in enumerate(top3_idx, start=1):
            disease = self.le.inverse_transform([idx])[0]
            confidence = round(proba[idx] * 100, 2)
            results.append({"rank": rank, "disease": disease, "confidence": confidence})

        return results

def show_all_symptoms(feature_names: list):
    """Print all valid symptom names in columns."""
    cols = 3
    print("\n--- Available Symptoms ---")
    for i, s in enumerate(feature_names):
        end = "\n" if (i + 1) % cols == 0 else "  |  "
        print(f"{s:<40}", end=end)
    print("\n")

def symptom_checker_cli():
    """Interactive command-line symptom checker."""
    try:
        predictor = SymptomPredictor()
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        return
        
    print("\n" + "=" * 55)
    print("      SYMPTOM-TO-DISEASE CHECKER")
    print("=" * 55)
    print("Enter your symptoms one at a time.")
    print("Commands:  list = show all symptoms | done = predict | quit = exit\n")

    while True:
        entered_symptoms = []

        while True:
            user_input = input("Enter symptom (or 'done'/'list'/'quit'): ").strip().lower()

            if user_input == "quit":
                print("Goodbye!")
                return

            elif user_input == "list":
                show_all_symptoms(predictor.feature_names)
                continue

            elif user_input == "done":
                if not entered_symptoms:
                    print("⚠  Please enter at least one symptom first.\n")
                    continue
                break

            elif user_input == "":
                continue

            else:
                if user_input in predictor.feature_names:
                    if user_input not in entered_symptoms:
                        entered_symptoms.append(user_input)
                        print(f"  ✔ Added: {user_input}  (total: {len(entered_symptoms)} symptoms)")
                    else:
                        print(f"  ℹ Already added: {user_input}")
                else:
                    suggestions = [s for s in predictor.feature_names if user_input in s][:3]
                    print(f"  ✗ '{user_input}' not found.", end="")
                    if suggestions:
                        print(f"  Did you mean: {suggestions}?")
                    else:
                        print(" Type 'list' to see all symptoms.")

        # Predict
        print(f"\n📋 Symptoms entered: {', '.join(entered_symptoms)}")
        predictions = predictor.predict_top3(entered_symptoms)

        print("\n" + "─" * 55)
        print("  🏥 TOP-3 PROBABLE DIAGNOSES")
        print("─" * 55)
        for p in predictions:
            bar = "█" * int(p["confidence"] / 4)
            medal = ["🥇", "🥈", "🥉"][p["rank"] - 1]
            print(f"  {medal}  {p['disease']:<44}  {p['confidence']:>6.2f}%")
            print(f"       {bar}")

        print("\n⚕  DISCLAIMER: This is a preliminary tool only.")
        print("   Please consult a qualified doctor for diagnosis.\n")

        again = input("Check another patient? (yes/no): ").strip().lower()
        if again not in ("yes", "y"):
            print("Thank you for using the Symptom Checker. Stay healthy!")
            break

if __name__ == "__main__":
    symptom_checker_cli()
