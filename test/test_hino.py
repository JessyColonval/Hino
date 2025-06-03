"""
Written by Jessy Colonval.
"""
from unittest import TestCase
from pandas import DataFrame
from numpy import arange
from src.hino import Hino


class TestHino(TestCase):

    @classmethod
    def setUpClass(cls):
        data = {
            "a0": [1.4, 1.3, 1.5, 1.7, 1.4, 4.5, 6.3, 5.8, 6.1, 5.1, 4.1,
                   4.5, 3.9, 4.8, 4.0],
            "a1": [0.2, 0.2, 0.2, 0.4, 0.3, 1.7, 1.8, 1.8, 2.5, 2.0, 1.0,
                   1.5, 1.1, 1.8, 1.3],
            "a2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.2, 4.8, 5.0, 5.1, 4.9],
            "a3": [2.0, 2.3, 1.9, 2.1, 1.9, 3.0, 3.2, 2.9, 2.9, 3.1, 4.0,
                   4.0, 4.0, 4.0, 4.0],
            "class": [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1],
            }
        cls.__dataset = DataFrame(data)

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
        cls.__bhv_distrib = [
            [{0: 1}, {0: 2}, {0: 2}, {1: 1}, {1: 1}, {1: 1}, {2: 1, 1: 1},
             {2: 1, 1: 1}, {2: 3}],
            [{0: 3}, {0: 2}, {1: 1}, {1: 1}, {1: 2}, {2: 1}, {}, {2: 4, 1: 1}],
            [{0: 5, 2: 5}, {1: 2}, {1: 3}],
            [{0: 2}, {0: 1}, {0: 2}, {}, {2: 2}, {2: 2}, {2: 1}, {1: 5}]
        ]

    def test_limit(self):
        """
        Verifies if the default tolerance limit is correct.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        self.assertEqual(0, model.limit)

    def test_set_limit(self):
        """
        Verifies if the modification of the tolerance limit work correctly.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        for limit in range(0, 100):
            model.set_limit(limit)
            self.assertEqual(limit, model.limit)

    def test_set_limit_negative(self):
        """
        Verifies if the modification of the tolerance limit raises an exception
        when a negative number is given.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        for limit in range(-100, 0):
            self.assertRaises(ValueError, model.set_limit, limit)

    def test_n_quantiles(self):
        """
        Verifies if the default number of quantiles is correct.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        self.assertEqual(4, model.n_quantiles)

    def test_set_n_quantiles(self):
        """
        Verifies if the modification of the number of quantiles work correctly.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        for n_quantiles in range(2, 100):
            model.set_n_quantiles(n_quantiles)
            self.assertEqual(n_quantiles, model.n_quantiles)

    def test_set_n_quantiles_negative(self):
        """
        Verifies if the modification of the number of quantiles raises an
        exception when a number lower than or equal to 1 is given.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        for n_quantiles in range(-100, 0):
            self.assertRaises(ValueError, model.set_n_quantiles, n_quantiles)

    def test_set_n_quantiles_is_zero(self):
        """
        Verifies if the modification of the number of quantiles raises an
        exception when 0 is given.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        self.assertRaises(ValueError, model.set_n_quantiles, 0)

    def test_set_n_quantiles_is_one(self):
        """
        Verifies if the modification of the number of quantiles raises an
        exception when 1 is given.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        self.assertRaises(ValueError, model.set_n_quantiles, 1)

    def test_n_points(self):
        """
        Verifies if the number of points retrieves is correct.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        self.assertEqual(15, model.n_points)

    def test_limit_estimator(self):
        """
        Verifies if the static function that calculate an estimation of the
        tolerance limit is correct.
        """
        for n_attr in range(1, 7):
            for n_cls in range(2, 4):
                self.assertEqual(0, Hino.limit_estimator(n_cls, n_attr))
        for n_attr in range(20, 30):
            self.assertEqual(1, Hino.limit_estimator(2, n_attr))

    def test_limit_estimator_wrong_n_attributes(self):
        """
        Verifies that the static function 'limit_estimator' throws an
        exception when the number of contextual attributes is less than or
        equal to 1.
        """
        self.assertRaises(ValueError, Hino.limit_estimator, 0, 2)
        self.assertRaises(ValueError, Hino.limit_estimator, -1, 2)
        self.assertRaises(ValueError, Hino.limit_estimator, -10, 2)

    def test_limit_estimator_wrong_n_classes(self):
        """
        Verifies that the static function 'limit_estimator' throws an
        exceptioon when the number of behavioral values is less than or equal
        to 2.
        """
        self.assertRaises(ValueError, Hino.limit_estimator, 1, 1)
        self.assertRaises(ValueError, Hino.limit_estimator, 1, 0)
        self.assertRaises(ValueError, Hino.limit_estimator, 1, -1)
        self.assertRaises(ValueError, Hino.limit_estimator, 1, -10)

    def test_set_max_percentage_outliers_zero(self):
        """
        Verifies if the initialization of the maximum percentage of allowed
        outliers raises an exception with 0.0 as parameter.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        self.assertRaises(ValueError, model.set_max_percent_outliers_detected,
                          0.0)

    def test_set_max_percentage_outliers_one(self):
        """
        Verifies if the initialization of the maximum percentage of allowed
        outliers raises an exception with 1.0 as parameter.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        self.assertRaises(ValueError, model.set_max_percent_outliers_detected,
                          1.0)

    def test_set_max_percentage_outliers_negative(self):
        """
        Verifies if the initialization of the maximum percentage of allowed
        outliers raises an exception with a negative number as parameter.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        for p in arange(-100, 0, 0.1):
            self.assertRaises(ValueError,
                              model.set_max_percent_outliers_detected,
                              p)

    def test_set_max_percentage_outliers_greater_than_one(self):
        """
        Verifies if the initialization of the maximum percentage of allowed
        outliers raises an exception with a positive number greater than 1.0
        as parameter.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        for p in arange(1.1, 100.0, 0.1):
            self.assertRaises(ValueError,
                              model.set_max_percent_outliers_detected,
                              p)

    def test_isolation_score(self):
        """
        Verifies if the computation of isolation scores works correctly with
        the default number of quantile.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        i_pts_qtils = [
            [[0, 1, 2, 4], [3, 12, 14], [5, 6, 7, 8, 9, 10, 11, 13]],
            [[0, 1, 2, 4], [3, 10, 12], [5, 6, 7, 8, 9, 11, 13, 14]],
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]],
            [[0, 2, 3, 4], [1, 7, 8], [5, 6, 9, 10, 11, 12, 13, 14]]
        ]
        bhv_distrib = [
            [{0: 4}, {0: 1, 2: 2}, {1: 5, 2: 3}],
            [{0: 4}, {0: 1, 2: 2}, {1: 5, 2: 3}],
            [{0: 5, 1: 5, 2: 5}],
            [{0: 4}, {0: 1, 1: 2}, {1: 3, 2: 5}]
        ]
        actual = model._Hino__n_cdts_failed(i_pts_qtils, bhv_distrib)
        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertListEqual(expected, actual)

    def test_isolation_score_w_10_quantiles(self):
        """
        Verifies if the computation of isolation scores works correctly with
        a custom number of quantile.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        model.set_n_quantiles(10)
        actual = model._Hino__n_cdts_failed(self.__i_pts_qtils,
                                            self.__bhv_distrib)
        expected = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
        self.assertListEqual(expected, actual)

    def test_is_outliers_no_tolerance(self):
        """
        Verifies if outlier decision works correctly when tolerance is limited
        to 0.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        isolation = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
        actual = model._Hino__is_outliers(isolation, 0)
        expected = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
        self.assertListEqual(expected, actual)

    def test_is_outliers_enough_tolerance(self):
        """
        Verifies if outlier decision works correctly when tolerance is limited
        to 1.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        isolation = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
        actual = model._Hino__is_outliers(isolation, 1)
        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertListEqual(expected, actual)

    def test_limit_review_exact_number_of_allowed_outliers(self):
        """
        Verifies if the tolerance limit increases by 1 when the maximum
        percentage of outlier allowed is exactly equal to the percentage of
        outlier detected.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        isolation = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
        model.set_max_percent_outliers_detected(0.4)
        self.assertEqual(0, model.limit)
        model._Hino__limit_review(isolation)
        self.assertEqual(1, model.limit)

    def test_limit_review_too_much_outliers(self):
        """
        Verifies if the tolerance limit increases by 1 when the maximum
        percentage of outlier allowed is lower than the percentage of
        outlier detected.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        isolation = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
        model.set_max_percent_outliers_detected(0.3)
        self.assertEqual(0, model.limit)
        model._Hino__limit_review(isolation)
        self.assertEqual(1, model.limit)

    def test_limit_review_enough_outliers(self):
        """
        Verifies if the tolerance limit doesn't increases when the maximum
        percentage of outlier allowed is higher than the percentage of
        outlier detected.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        isolation = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
        model.set_max_percent_outliers_detected(0.5)
        self.assertEqual(0, model.limit)
        model._Hino__limit_review(isolation)
        self.assertEqual(0, model.limit)

    def test_limit_review_wo_a_bound(self):
        """
        Verifies that the limit review raises an exception when the maximal
        percent of allowed outliers is not set.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        isolation = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
        self.assertRaises(ValueError, model._Hino__limit_review, isolation)

    def test_update_isolation(self):
        """
        Verifies that isolation scores change for the point in the quantile
        where its behavioral values are missing.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        isolation = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
        qtil = [0, 4]
        absent_bhv = [0]
        model._Hino__update_isolation(isolation, absent_bhv, qtil)
        actual = model._Hino__is_outliers(isolation, 0)
        expected = [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
        self.assertListEqual(expected, actual)

    def test_update_isolation_no_absent_bhv(self):
        """
        Verifies that isolation scores do not change when no behavioral values
        are missing.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        isolation = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
        qtil = [0, 4]
        absent_bhv = []
        model._Hino__update_isolation(isolation, absent_bhv, qtil)
        actual = model._Hino__is_outliers(isolation, 0)
        expected = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
        self.assertListEqual(expected, actual)

    def test_update_isolation_other_absent_bhv(self):
        """
        Verifies that isolation scores do not change when the behavioral values
        missing are not the value of the points in the quantile.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        isolation = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
        qtil = [0, 4]
        absent_bhv = [1, 2]
        model._Hino__update_isolation(isolation, absent_bhv, qtil)
        actual = model._Hino__is_outliers(isolation, 0)
        expected = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
        self.assertListEqual(expected, actual)

    def test_fit(self):
        """
        Verifies if the Hino's detection work correctly.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        model.set_n_quantiles(10)
        actual = model.fit()
        expected = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
        self.assertListEqual(expected, actual)

    def test_fit_too_much_outliers(self):
        """
        Verifies that the tolerance limit increases and that the outliers
        detected are smaller when a percentage of allowed outliers is lower
        than the outliers detected the first time.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        model.set_n_quantiles(10)
        model.set_max_percent_outliers_detected(0.2)
        actual = model.fit()
        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertListEqual(expected, actual)
        self.assertEqual(1, model.limit)

    def test_fit_enough_outliers(self):
        """
        Verifies that the tolerance limit and detected outliers do not change
        when a percentage of allowed outliers is higher than the outliers
        detected the first time.
        """
        model = Hino(self.__dataset, ["a0", "a1", "a2", "a3"], "class")
        model.set_n_quantiles(10)
        model.set_max_percent_outliers_detected(0.5)
        actual = model.fit()
        expected = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
        self.assertListEqual(expected, actual)
        self.assertEqual(0, model.limit)
