import unittest
import os
from src.models.predict import SymptomPredictor

class TestPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize the predictor (loads the final model from artifacts)
        cls.predictor = SymptomPredictor()

    def test_predictor_loaded(self):
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.le)
        self.assertGreater(len(self.predictor.feature_names), 0)

    def test_prediction_output_format(self):
        # Predict with a common symptom
        test_symptoms = ["fever", "cough"]
        predictions = self.predictor.predict_top3(test_symptoms)
        
        # Verify predictions structure
        self.assertEqual(len(predictions), 3)
        for p in predictions:
            self.assertIn("rank", p)
            self.assertIn("disease", p)
            self.assertIn("confidence", p)
            self.assertTrue(0.0 <= p["confidence"] <= 100.0)

    def test_invalid_symptoms(self):
        # Predict with unrecognized symptoms
        predictions = self.predictor.predict_top3(["invalid_symptom_xyz"])
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["disease"], "No valid symptoms entered")
        self.assertEqual(predictions[0]["confidence"], 0.0)
