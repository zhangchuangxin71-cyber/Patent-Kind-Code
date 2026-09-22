import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_module_ablation.py"
SPEC = importlib.util.spec_from_file_location("summarize_module_ablation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ModuleAblationReportingTests(unittest.TestCase):
    def test_paired_reports_positive_negative_and_tied_seeds(self):
        result = MODULE.paired([2.0, 1.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 5.0, 4.0])
        self.assertEqual(result["positive_seed_count"], 2)
        self.assertEqual(result["negative_seed_count"], 2)
        self.assertEqual(result["tied_seed_count"], 1)
        self.assertEqual(result["non_tied_seed_count"], 4)
        self.assertEqual(result["total_seed_count"], 5)
        self.assertAlmostEqual(result["one_sided_exact_sign_p"], 0.6875)

    def test_shuffle_inference_uses_five_seed_level_differences(self):
        real = [0.5, 0.6, 0.7, 0.8, 0.9]
        shuffle_means = [0.4, 0.5, 0.6, 0.7, 0.8]
        result = MODULE.paired(real, shuffle_means)
        self.assertEqual(result["total_seed_count"], 5)
        self.assertEqual(result["positive_seed_count"], 5)
        self.assertAlmostEqual(result["one_sided_exact_sign_p"], 0.03125)


if __name__ == "__main__":
    unittest.main()
