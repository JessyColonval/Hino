"""
Written by Jessy Colonval.
"""
from unittest import TestCase
from src.quantile import Quantile


class TestQuantile(TestCase):
    """
    Unit tests to verify the proper execution of the Quantile object's static
    functions.
    All these tests are run on the same example dataset.
    """

    @classmethod
    def setUpClass(cls):
        # The number of points in the dataset.
        cls.__n_pts = 15

        # The dataset.
        cls.__data = [
            [1.4, 1.3, 1.5, 1.7, 1.4, 4.5, 6.3, 5.8, 6.1, 5.1, 4.1, 4.5, 3.9,
             4.8, 4.0],
            [0.2, 0.2, 0.2, 0.4, 0.3, 1.7, 1.8, 1.8, 2.5, 2.0, 1.0, 1.5, 1.1,
             1.8, 1.3],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.2, 4.8, 5.0, 5.1, 4.9],
            [2.0, 2.3, 1.9, 2.1, 1.9, 3.0, 3.2, 2.9, 2.9, 3.1, 4.0, 4.0, 4.0,
             4.0, 4.0]
        ]
        cls.__bhvs = [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1]

        # The separation of the values of each contextual attribute in the
        # dataset for the construction of 10 quantiles.
        cls.__qtils = [
            [1.3, 1.4, 1.48, 2.14, 3.96, 4.1, 4.5, 4.74, 5.24, 6.3],
            [0.2, 0.28, 0.52, 1.06, 1.3, 1.58, 1.78, 1.8, 2.5],
            [0.0, 3.84, 4.92, 5.2],
            [1.9, 1.94, 2.08, 2.42, 2.9, 3.0, 3.14, 3.84, 4.0]
        ]

        # The separation of dataset points into 10 quantiles for each
        # contextual attribute.
        cls.__i_pts_qtils = [
            [[1], [0, 4], [2, 3], [12], [14], [10], [5, 11], [9, 13],
             [6, 7, 8]],
            [[0, 1, 2], [3, 4], [10], [12], [11, 14], [5], [],
             [6, 7, 8, 9, 13]],
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [11, 14], [10, 12, 13]],
            [[2, 4], [0], [1, 3], [], [7, 8], [5, 9], [6],
             [10, 11, 12, 13, 14]]
        ]

    def test_quantiles(self):
        """
        Verifies that the 'quantiles' static function correctly constructs the
        threshold values separating each quantile for all contextual
        attributes in the dataset.
        """
        actual = Quantile.quantiles(self.__data, 10)
        expected = self.__qtils
        self.assertEqual(len(actual), len(expected))
        for i, expected_row in enumerate(expected):
            self.assertEqual(len(actual[i]), len(expected_row))
            for j, expected_val in enumerate(expected_row):
                self.assertAlmostEqual(actual[i][j], expected_val)

    def test_quantiles_wrong_n_qtils(self):
        """
        Verifies that the 'quantiles' static function does not allow quantile
        separation when the number chosen is less than or equal to 1.
        """
        self.assertRaises(ValueError, Quantile.quantiles, self.__data, 1)
        self.assertRaises(ValueError, Quantile.quantiles, self.__data, 0)
        self.assertRaises(ValueError, Quantile.quantiles, self.__data, -1)
        self.assertRaises(ValueError, Quantile.quantiles, self.__data, -10)

    def test_points_quantiles_all_points_are_presents(self):
        """
        Verifies that the 'points_per_quantiles' static function assigns all
        points to a quantile for each contextual attribute.
        """
        actual = Quantile.points_per_quantiles(self.__data, self.__qtils)
        expected = all(
            all(
                any(i in qtil for qtil in ctx_qtils)
                for i in range(0, self.__n_pts)
            )
            for ctx_qtils in actual
        )
        self.assertTrue(expected)

    def test_points_per_quantiles(self):
        """
        Verifies that the 'points_per_quantiles' static function that correctly
        splits all points in the dataset into the correct quantiles (i.e.
        according to their contextual values) for each contextual attribute.
        """
        actual = Quantile.points_per_quantiles(self.__data, self.__qtils)
        expected = self.__i_pts_qtils
        self.assertListEqual(expected, actual)

    def test_quantiles_distribution(self):
        """
        Verifies that the static function 'quantiles_distribution' correctly
        computes the number of elements of each behavioral value for each
        quantile of all contextual attributes in the dataset.
        """
        actual = Quantile.quantiles_distribution(self.__bhvs,
                                                 self.__i_pts_qtils)
        expected = [
            [{0: 1}, {0: 2}, {0: 2}, {1: 1}, {1: 1}, {1: 1}, {2: 1, 1: 1},
             {2: 1, 1: 1}, {2: 3}],
            [{0: 3}, {0: 2}, {1: 1}, {1: 1}, {1: 2}, {2: 1}, {}, {2: 4, 1: 1}],
            [{0: 5, 2: 5}, {1: 2}, {1: 3}],
            [{0: 2}, {0: 1}, {0: 2}, {}, {2: 2}, {2: 2}, {2: 1}, {1: 5}]
        ]
        self.assertListEqual(expected, actual)

    def test_quantiles_distribution_not_engouh_behavioral_values(self):
        """
        Verifies that the static function 'quantiles_distribution' raises an
        exception when there are fewer behavioral values than there are
        points.
        """
        for i in range(0, self.__n_pts-1):
            self.assertRaises(ValueError, Quantile.quantiles_distribution,
                              self.__bhvs[:i], self.__i_pts_qtils)

    def test_quantiles_distribution_too_much_behavioral_values(self):
        """
        Verifies that the static function 'quantiles_distribution' raises an
        exception when there are higher behavioral values than there are
        points.
        """
        bhvs = self.__bhvs + [1]
        self.assertRaises(ValueError, Quantile.quantiles_distribution, bhvs,
                          self.__i_pts_qtils)
