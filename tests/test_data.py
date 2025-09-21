from random import random
import tempfile
from unittest import TestCase

import numpy as np
import pandas as pd
from ocpy import Data
from ocpy.errors import LengthCheckError


class TestData(TestCase):
    def setUp(self):
        self.DATA = Data(np.linspace(2460925.5, 2460935.5, 200).tolist())

    def test_from_file(self):
        df = pd.DataFrame(
            {
                "time": np.linspace(2460925.5, 2460935.5, 200).tolist(),
                "error": np.random.random(200).tolist(),
                "weight": np.random.random(200).tolist(),
                "type": np.random.choice([True, False], size=200),
                "label": np.random.choice(["CCD", "Phot"], size=200),
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=True, mode="w", newline="", encoding="utf-8") as tmp:
            df.to_csv(tmp.name, index=False)
            data = Data.from_file(
                tmp.name,
                {
                    "minimum_time": "time",
                    "minimum_time_error": "error",
                    "weights": "weight",
                    "minimum_type": "type",
                    "labels": "label",
                }
            )

            self.assertEqual(len(data), len(df))

    def test_fill_errors_list(self):
        new_data = self.DATA.fill_errors([1] * len(self.DATA))
        self.assertEqual(
            new_data["minimum_time_error"].isna().sum(),
            0
        )

    def test_fill_errors_list_no_override(self):
        new_data = self.DATA.fill_errors([1, None] * (len(self.DATA) // 2))

        new_data = new_data.fill_errors([2] * len(self.DATA))
        self.assertEqual(
            new_data["minimum_time_error"].isna().sum(),
            0
        )
        self.assertEqual(
            new_data["minimum_time_error"].sum(),
            3 * len(new_data) // 2
        )

    def test_fill_errors_list_with_override(self):
        new_data = self.DATA.fill_errors([1, None] * (len(self.DATA) // 2))

        new_data = new_data.fill_errors([2] * len(self.DATA), override=True)
        self.assertEqual(
            new_data["minimum_time_error"].isna().sum(),
            0
        )
        self.assertEqual(
            new_data["minimum_time_error"].sum(),
            2 * len(new_data)
        )

    def test_fill_errors_list_wrong_length(self):
        with self.assertRaises(LengthCheckError):
            self.DATA.fill_errors([1] * (len(self.DATA) - 1))

    def test_fill_errors_numeric(self):
        new_data = self.DATA.fill_errors(1)
        self.assertEqual(
            new_data["minimum_time_error"].isna().sum(),
            0
        )

    def test_fill_errors_numeric_no_override(self):
        new_data = self.DATA.fill_errors([1, None] * (len(self.DATA) // 2))

        new_data = new_data.fill_errors(2)
        self.assertEqual(
            new_data["minimum_time_error"].isna().sum(),
            0
        )
        self.assertEqual(
            new_data["minimum_time_error"].sum(),
            3 * len(new_data) // 2
        )

    def test_fill_errors_numeric_with_override(self):
        new_data = self.DATA.fill_errors([1, None] * (len(self.DATA) // 2))

        new_data = new_data.fill_errors(2, override=True)
        self.assertEqual(
            new_data["minimum_time_error"].isna().sum(),
            0
        )
        self.assertEqual(
            new_data["minimum_time_error"].sum(),
            2 * len(new_data)
        )

    def test_fill_weights_list(self):
        new_data = self.DATA.fill_weights([1] * len(self.DATA))
        self.assertEqual(
            new_data["weights"].isna().sum(),
            0
        )

    def test_fill_weights_list_no_override(self):
        new_data = self.DATA.fill_weights([1, None] * (len(self.DATA) // 2))

        new_data = new_data.fill_weights([2] * len(self.DATA))
        self.assertEqual(
            new_data["weights"].isna().sum(),
            0
        )
        self.assertEqual(
            new_data["weights"].sum(),
            3 * len(new_data) // 2
        )

    def test_fill_weight_list_with_override(self):
        new_data = self.DATA.fill_weights([1, None] * (len(self.DATA) // 2))

        new_data = new_data.fill_weights([2] * len(self.DATA), override=True)
        self.assertEqual(
            new_data["weights"].isna().sum(),
            0
        )
        self.assertEqual(
            new_data["weights"].sum(),
            2 * len(new_data)
        )

    def test_fill_weights_list_wrong_length(self):
        with self.assertRaises(LengthCheckError):
            _ = self.DATA.fill_weights([1] * (len(self.DATA) - 1))

    def test_fill_weights_numeric(self):
        new_data = self.DATA.fill_weights(1)
        self.assertEqual(
            new_data["weights"].isna().sum(),
            0
        )

    def test_fill_weights_numeric_no_override(self):
        new_data = self.DATA.fill_weights([1, None] * (len(self.DATA) // 2))

        new_data = new_data.fill_weights(2)
        self.assertEqual(
            new_data["weights"].isna().sum(),
            0
        )
        self.assertEqual(
            new_data["weights"].sum(),
            3 * len(new_data) // 2
        )

    def test_fill_weights_numeric_with_override(self):
        new_data = self.DATA.fill_weights([1, None] * (len(self.DATA) // 2))

        new_data = new_data.fill_weights(2, override=True)
        self.assertEqual(
            new_data["weights"].isna().sum(),
            0
        )
        self.assertEqual(
            new_data["weights"].sum(),
            2 * len(new_data)
        )

    def test_calculate_weights(self):
        the_error = random() * 2
        new_data = self.DATA.fill_errors(the_error)
        new_weighted_data = new_data.calculate_weights()
        self.assertEqual(new_weighted_data["weights"].sum(), sum([1/np.pow(the_error, 2)] * len(new_data)))

    def test_calculate_weight_with_no_error(self):
        with self.assertRaises(ValueError):
            _ = self.DATA.calculate_weights()

    def test_calculate_weights_custom_function(self):
        the_error = random() * 2
        new_data = self.DATA.fill_errors(the_error)
        def custom_function(data):
            return pd.Series([1] * len(data))

        new_weighted_data = new_data.calculate_weights(method=custom_function)
        self.assertEqual(new_weighted_data["weights"].sum(), len(new_data))

    def test_calculate_oc(self):
        oc = self.DATA.calculate_oc(
            float(np.mean(np.diff(self.DATA["minimum_time"]))),
            float(self.DATA[0]["minimum_time"])
        )
        self.assertEqual(oc["oc"].sum(), 0)

    def test_merge(self):
        data = Data(np.linspace(2460965.5, 2460975.5, 50).tolist())
        merged_data = data.merge(self.DATA)
        self.assertEqual(
            len(merged_data),
            len(data) + len(self.DATA)
        )

        for each in data:
            self.assertIn(each["minimum_time"][0], merged_data["minimum_time"].tolist())
        for each in self.DATA:
            self.assertIn(each["minimum_time"][0], merged_data["minimum_time"].tolist())

    def test_group_by_none(self):
        grouped_data = self.DATA.group_by("labels")
        self.assertEqual(len(grouped_data), 1)
        self.assertEqual(
            len(grouped_data[0]), len(self.DATA)
        )

    def test_group_by(self):
        data = Data(
            np.linspace(2460925.5, 2460935.5, 200).tolist(),
            minimum_time_error=np.random.random(200).tolist(),
            weights=np.random.random(200).tolist(),
            minimum_type=np.random.choice([True, False], size=200),
            labels=np.random.choice(["CCD", "Phot"], size=200),
            ecorr=np.random.random(200).tolist(),
        )
        grouped_data = data.group_by("labels")
        self.assertEqual(len(grouped_data), len(data["labels"].unique()))
        self.assertEqual(
            sum(len(each) for each in grouped_data),
            len(data)
        )
        for each_group in grouped_data:
            the_label = each_group[0]["labels"][0]
            for each in each_group:
                self.assertIn(each["minimum_time"][0], data[data["labels"] == the_label]["minimum_time"].tolist())
