"""
Written by Jessy Colonval.
"""
from typing import List, Dict, Any
from collections import Counter
import numpy as np


class Quantile():
    """
    An object that contains only static functions for constructing quantiles
    used by the Hino algorithm.
    """

    @staticmethod
    def quantiles(data: List[List[float]], n_qtils: int) -> List[List[float]]:
        """
        For a desired number of quantiles, calculates all the threshold values
        that separate them for each contextual attribute of a given data set.

        Parameters
        ----------
        data: List[List[float]]
            Contextual values of the dataset.
        n_qtils: int
            The desired number of quantiles.

        Return
        ------
        List[List[float]]
            The threshold values separating each quantile for all contextual
            attributes.
            Duplicates are removed so that points lying between the same
            thresholds end up in the same quantile instead of being arbitrarily
            separated.
        """
        if n_qtils <= 1:
            raise ValueError("The number of quantiles must be strictly ",
                             "greater than 1.")

        # Calculates the percent of points present in each quantile.
        step = 1.0 / n_qtils

        # Builds a list of double representing the different percentage point
        # thresholds (between 0.0 and 1.0) contained in each quantile.
        # The value 1.0 is added manually at the end to avoid precision
        # problems.
        ptils = np.arange(0.0, 1.0-step, step).tolist()
        ptils.append(1.0)

        # Calculates the threshold values between each quantile for all
        # contextual attributes.
        qtils = np.quantile(data, ptils, axis=1)

        # Invert rows and columns, then remove duplicates from all rows.
        return [np.unique(row) for row in qtils.T]

    @staticmethod
    def points_per_quantiles(data: List[List[float]],
                             ctxs_qtils: List[List[float]],
                             lower_epsilon: float = None,
                             upper_epsilon: float = None
                             ) -> List[List[List[int]]]:
        """
        Divides the dataset into several quantiles for each of these contextual
        attributes.

        Parameters
        ----------
        data: List[List[float]]
            Contextual values of the dataset.
        ctxs_qtils: List[List[float]]
            The threshold values separating each quantile for all contextual
            attributes.
        lower_epsilon: float
           Lower epsilon to be applied to take into consideration points
           located in the previous quantile when they are “too close” to
           the next quantile.
           It is equal to the difference between the two thresholds multiplied
           by this epsilon, i.e. epsilon * (t_{i} - t_{i-1})
        upper_epsilon: float
            Upper epsilon to be applied in order to take into consideration
            points located in the next quantile when they are “too close”
            to the previous quantile.
            It is equal to the difference between the two thresholds multiplied
            by this epsilon, i.e. epsilon * (t_{i+1} - t_{i})

        Return
        ------
        List[List[List[int]]]
            A principal list, with the size of the number of contextual
            attributes, containing the indices of the points separated
            according to the threshold values supplied as parameters.
        """
        result = []

        # Iterates all contextual attributes of the dataset.
        for i_col, qtils in enumerate(ctxs_qtils):
            result.append([])

            # Iterates all quantiles of the current contextual attributes.
            for i in range(0, len(qtils) - 1):
                # Current threshold.
                qi = qtils[i]

                # The minimum bound is equal to the current threshold, by
                # default.
                # When the lower epsilon is given, then this bound is the
                # current threshold, from which a percentage of the difference
                # between the two successive thresholds is subtracted.
                min_val = qi
                if lower_epsilon is not None and i > 0:
                    min_val -= lower_epsilon * (qi - qtils[i-1])

                # The next threshold.
                qiplus1 = qtils[i+1]

                # Tha maximum bound is equal to the current threshold, by
                # default.
                # When the upper epsilon is given, then this bound is the
                # current threshold, from which a percentage of the difference
                # between the two successive thresholds is added.
                max_val = qiplus1
                if upper_epsilon is not None and i < len(qtils) - 2:
                    max_val += upper_epsilon * (qtils[i+2] - qiplus1)

                # Includes points with a contextual value equal to the maximum
                # threshold when it's the last quantile.
                if i == len(qtils) - 2:
                    indices = [
                        i_pts
                        for i_pts, ctx_val in enumerate(data[i_col])
                        if min_val <= ctx_val <= max_val
                    ]

                # By default, includes points with a contextual value between
                # the lower bound (include) and the upper bound (not include).
                else:
                    indices = [
                        i_pts
                        for i_pts, ctx_val in enumerate(data[i_col])
                        if min_val <= ctx_val < max_val
                    ]

                # Adds quantile-separated point indices according to the
                # threshold values of the current contextual attribute.
                result[i_col].append(indices)

        return result

    @staticmethod
    def quantiles_distribution(behaviors: List[Any],
                               qtils_pts: List[List[List[int]]]
                               ) -> List[List[Dict[Any, int]]]:
        """
        Gives the number of behavioral values present in each quantile of each
        contextual attribute.

        Parameters
        ----------
        behaviors: List[Any]
            The list of behavioral values of the points in the dataset.
            The order must be respected, i.e. the nth behavioral value is
            associated with the nth point.
            In addition, these values can be represented as any mutable type.
        qtils_pts: List[List[List[int]]]
            The indices of the points present in each quantile for each
            contextual attribute of the dataset.
            It is assumed that all points in the dataset are present in at
            least one quantile per contextual attribute.

        Return
        ------
        List[List[Dict[Any, int]]]
            For each contextual attribute, give the list of behavioral values
            present in each quantile.
            Thus, each quantile is associated with a mapping that matches a
            behavioral value (of type Any) with its number of occurrences.
        """
        # Computes the largest index present in the given quantiles.
        # No need to search for all contextual attributes, as all points must
        # be present in at least one quantile for each attribute.
        n_pts = max(
            i_pts
            for qtil in qtils_pts[0]
            for i_pts in qtil
        )

        # Throw an exception if there are not as many behavioral values as
        # there are points.
        if len(behaviors)-1 != n_pts:
            raise ValueError("The number of behavioral values must be equal ",
                             "to the number of points in the quantiles.")

        return [
            [
                dict(Counter(behaviors[i_pts] for i_pts in i_pts_qtils))
                for i_pts_qtils in i_pts_col
            ]
            for i_pts_col in qtils_pts
        ]
