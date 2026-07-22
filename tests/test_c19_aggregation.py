import unittest

from experiments.bibm_smile.aggregate_results import aggregate, fmt_metric


class C19AggregationTest(unittest.TestCase):
    def test_three_seed_summary_uses_sample_standard_deviation(self):
        values = [81.03, 83.87, 82.44]
        self.assertEqual(fmt_metric(values), r"82.45\sm1.42")
        rows = [
            {
                "status": "ok",
                "dataset": "c19",
                "variant_name": "SMILE-Full",
                "variant": "smart-smile-lean",
                "metric": "auprc",
                "value": value,
            }
            for value in values
        ]
        self.assertAlmostEqual(aggregate(rows)[0]["std"], 1.4200117370406973)


if __name__ == "__main__":
    unittest.main()

