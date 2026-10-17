"""Smoke tests for the saved NEET model's inference interface.

Run from the repository root with: python -m unittest discover -s tests
"""
import unittest
from pathlib import Path

import pandas as pd

from model.inference import score, load_bundle


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "neet_dataset.csv"


class ModelInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle()
        cls.input_data = pd.read_csv(DATA_PATH).head(20)
        cls.features = cls.input_data[cls.bundle["features"]]
        cls.result = score(cls.input_data, cls.bundle)

    def test_scores_one_result_per_learner(self):
        self.assertEqual(len(self.result), len(self.input_data))
        self.assertEqual(
            self.result["learner_id"].tolist(),
            self.input_data["learner_id"].tolist(),
        )

    def test_result_has_expected_columns(self):
        self.assertEqual(
            self.result.columns.tolist(),
            [
                "learner_id",
                "risk_score",
                "risk_level",
                "top_factor_1",
                "top_factor_2",
                "top_factor_3",
                "suggested_intervention",
            ],
        )

    def test_risk_scores_and_levels_are_valid(self):
        self.assertTrue(self.result["risk_score"].between(0, 1).all())
        self.assertTrue(
            self.result["risk_level"].isin(["Low", "Medium", "High"]).all()
        )
        self.assertTrue(self.result["suggested_intervention"].ne("").all())

    def test_missing_feature_is_reported(self):
        missing_feature = self.bundle["features"][0]
        incomplete = self.features.drop(columns=[missing_feature])

        with self.assertRaisesRegex(ValueError, missing_feature):
            score(incomplete, self.bundle)


if __name__ == "__main__":
    unittest.main()
