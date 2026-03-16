import os
import tempfile
from random import randint
from unittest import TestCase

import numpy as np
import pandas as pd

from ocpy.oc import OC
from ocpy.oc_lmfit import OCLMFit
from tests.utils import N


class TestOC(TestCase):
    def setUp(self):
        self.n = 50
        self.cycle = np.arange(self.n, dtype=float).tolist()
        self.oc_vals = (np.random.normal(0, 0.01, self.n)).tolist()
        self.min_time = np.linspace(2460000.0, 2460050.0, self.n).tolist()
        self.errors = (np.full(self.n, 0.001)).tolist()
        self.weights = np.ones(self.n).tolist()
        self.labels = ["CCD"] * self.n
        self.min_type = [None] * self.n

        self.oc = OC(
            oc=self.oc_vals,
            minimum_time=self.min_time,
            minimum_time_error=self.errors,
            weights=self.weights,
            minimum_type=self.min_type,
            labels=self.labels,
            cycle=self.cycle,
        )

    # __len__
    def test_len(self):
        self.assertEqual(len(self.oc), self.n)

    # __str__
    def test_str(self):
        s = str(self.oc)
        self.assertIsInstance(s, str)
        self.assertGreater(len(s), 0)

    # __getitem__ string column
    def test_getitem_column(self):
        col = self.oc["oc"]
        self.assertEqual(len(col), self.n)

    # __getitem__ integer index
    def test_getitem_int(self):
        item = self.oc[0]
        self.assertIsInstance(item, OC)
        self.assertEqual(len(item), 1)

    # __getitem__ boolean mask
    def test_getitem_mask(self):
        mask = self.oc.data["cycle"] < 10
        subset = self.oc[mask]
        self.assertIsInstance(subset, OC)
        self.assertEqual(len(subset), int(mask.sum()))

    # __setitem__
    def test_setitem(self):
        new_labels = ["Phot"] * self.n
        self.oc["labels"] = new_labels
        self.assertTrue((self.oc.data["labels"] == "Phot").all())

    # from_file csv
    def test_from_file_csv(self):
        df = pd.DataFrame({
            "minimum_time": self.min_time,
            "minimum_time_error": self.errors,
            "weights": self.weights,
            "minimum_type": self.min_type,
            "labels": self.labels,
            "cycle": self.cycle,
            "oc": self.oc_vals,
        })

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".csv", delete=False, mode="w", newline="", encoding="utf-8"
            ) as tmp:
                tmp_name = tmp.name
                df.to_csv(tmp_name, index=False)

            loaded = OC.from_file(tmp_name)
            self.assertEqual(len(loaded), self.n)
        finally:
            if "tmp_name" in locals() and os.path.exists(tmp_name):
                os.unlink(tmp_name)

    # from_file with column mapping
    def test_from_file_column_mapping(self):
        df = pd.DataFrame({
            "time": self.min_time,
            "err": self.errors,
            "w": self.weights,
            "cycle": self.cycle,
            "oc": self.oc_vals,
        })

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".csv", delete=False, mode="w", newline="", encoding="utf-8"
            ) as tmp:
                tmp_name = tmp.name
                df.to_csv(tmp_name, index=False)

            loaded = OC.from_file(tmp_name, columns={"time": "minimum_time", "err": "minimum_time_error", "w": "weights"})
            self.assertEqual(len(loaded), self.n)
        finally:
            if "tmp_name" in locals() and os.path.exists(tmp_name):
                os.unlink(tmp_name)

    # from_file unsupported format
    def test_from_file_unsupported(self):
        with self.assertRaises(ValueError):
            OC.from_file("file.txt")

    # merge
    def test_merge(self):
        other = OC(
            oc=[0.0] * 10,
            minimum_time=np.linspace(2460100.0, 2460110.0, 10).tolist(),
            cycle=np.arange(50, 60, dtype=float).tolist(),
            weights=[1.0] * 10,
        )
        merged = self.oc.merge(other)
        self.assertEqual(len(merged), self.n + 10)

    # _equal_bins
    def test_equal_bins(self):
        edges = OC._equal_bins(self.oc.data, "cycle", 5)
        self.assertEqual(len(edges), 6)
        self.assertAlmostEqual(edges[0], min(self.cycle))
        self.assertAlmostEqual(edges[-1], max(self.cycle))

    # _smart_bins
    def test_smart_bins(self):
        bins = OC._smart_bins(self.oc.data, "cycle", 5, smart_bin_period=100.0)
        self.assertGreater(len(bins), 0)
        self.assertEqual(bins.shape[1], 2)

    def test_smart_bins_invalid_period(self):
        with self.assertRaises(ValueError):
            OC._smart_bins(self.oc.data, "cycle", 5, smart_bin_period=-1)

    # bin
    def test_bin_equal(self):
        binned = self.oc.bin(bin_count=5)
        self.assertLessEqual(len(binned), 5)
        self.assertGreater(len(binned), 0)

    def test_bin_smart(self):
        binned = self.oc.bin(
            bin_count=5,
            bin_style=lambda df, n: OC._smart_bins(df, "cycle", n, smart_bin_period=100.0),
        )
        self.assertGreater(len(binned), 0)

    def test_bin_missing_weights_nan(self):
        oc_obj = OC(
            oc=self.oc_vals,
            minimum_time=self.min_time,
            cycle=self.cycle,
            weights=None,
        )
        with self.assertRaises(ValueError):
            oc_obj.bin(bin_count=5)

    # calculate_oc
    def test_calculate_oc(self):
        ref_min = self.min_time[0]
        ref_per = float(np.mean(np.diff(self.min_time)))

        result = self.oc.calculate_oc(ref_min, ref_per)
        self.assertEqual(len(result), self.n)
        self.assertIn("oc", result.data.columns)

    def test_calculate_oc_returns_lmfit(self):
        ref_min = self.min_time[0]
        ref_per = float(np.mean(np.diff(self.min_time)))

        result = self.oc.calculate_oc(ref_min, ref_per, model_type="lmfit")
        self.assertIsInstance(result, OCLMFit)

    def test_calculate_oc_with_secondary(self):
        min_type = ["primary"] * 25 + ["secondary"] * 25
        oc = OC(
            oc=self.oc_vals,
            minimum_time=self.min_time,
            minimum_time_error=self.errors,
            weights=self.weights,
            minimum_type=min_type,
            labels=self.labels,
            cycle=self.cycle,
        )
        ref_min = self.min_time[0]
        ref_per = float(np.mean(np.diff(self.min_time)))

        result = oc.calculate_oc(ref_min, ref_per)
        self.assertEqual(len(result), self.n)

    # randomized
    def test_randomized_merge(self):
        for _ in range(N):
            n1 = randint(5, 50)
            n2 = randint(5, 50)
            oc1 = OC(
                oc=np.random.normal(0, 0.01, n1).tolist(),
                cycle=np.arange(n1, dtype=float).tolist(),
                weights=np.ones(n1).tolist(),
            )
            oc2 = OC(
                oc=np.random.normal(0, 0.01, n2).tolist(),
                cycle=np.arange(n1, n1 + n2, dtype=float).tolist(),
                weights=np.ones(n2).tolist(),
            )
            merged = oc1.merge(oc2)
            self.assertEqual(len(merged), n1 + n2)
