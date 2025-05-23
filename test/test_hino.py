"""
Written by Jessy Colonval.
"""
from unittest import TestCase
from pandas import DataFrame
from src.hino import Hino


class TestHino(TestCase):

    @classmethod
    def setUpClass(cls):
        data = {
            "attr0": [1.4, 1.3, 1.5, 1.7, 1.4, 4.5, 6.3, 5.8, 6.1, 5.1, 4.1,
                      4.5, 3.9, 4.8, 4.0],
            "attr1": [0.2, 0.2, 0.2, 0.4, 0.3, 1.7, 1.8, 1.8, 2.5, 2.0, 1.0,
                      1.5, 1.1, 1.8, 1.3],
            "attr2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.2, 4.8, 5.0, 5.1, 4.9],
            "attr3": [2.0, 2.3, 1.9, 2.1, 1.9, 3.0, 3.2, 2.9, 2.9, 3.1, 4.0,
                      4.0, 4.0, 4.0, 4.0],
            "class0": [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1],
            }
        dataset = DataFrame(data)
        cls.__model = Hino(dataset, ["attr0", "attr1", "attr2", "attr3"],
                           "class0")
        cls.__model.set_n_quantiles(10)

    def test_limit_estimator(self):
        actual = Hino.limit_estimator(4, 3)
        self.assertEqual(0, actual)

    def test_n_conditions_failed(self):
        actual = self.__model._Hino__n_cdts_failed()
        expected = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
        self.assertListEqual(expected, actual)
