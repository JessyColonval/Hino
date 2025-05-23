"""
Written by Jessy Colonval.
"""
from typing import List
from collections import Counter
from pandas import DataFrame
from numpy import log2
from src.quantile import Quantile


class Hino():
    """
    Object implementing the Hino algorithm presented in the article 'Hunting
    inside n-quantiles of outliers (Hino)' in 'The 29th Pacific-Asia Conference
    on Knowledge Discovery and Data Mining'.

    Attributes
    ----------
    __val_ctx: List[List[float]]
        Contextual dataset values in the form of a collection of lists for each
        contextual attribute.
    __val_bhv: List[int | str]
        Behavioral dataset values.
    __n_pts: int
        The number of points in the dataset.
    __distrib: Counter
        The number of points for each behavioral values.
    __limit: int
        The applied tolerance limit, which determines how many times a point
        must be isolated by contextual attribute before it is considered an
        outlier.
    __n_qtils: int
        The number of quantiles constructed for each contextual attribute.
    __max_p_outliers: float
        The percentage, between 0.0 and 1.0 (not include), of outliers allowed
        to keep the dataset consistent even after removing these outliers.
        If detection finds more, it will be restarted with a more restrictive
        tolerance limit until this constraint is satisfied.
    """

    __val_ctx: List[List[float]]
    __val_bhv: List[int | str]
    __n_pts: int
    __distrib: Counter
    __limit: int
    __n_qtils: int
    __max_p_outliers: float

    def __init__(self, dataset: DataFrame, col_contex: List[str],
                 col_behav: str) -> None:
        # Splits the contextual and the behavioral values.
        self.__val_ctx = [list(dataset[col]) for col in col_contex]
        self.__val_bhv = list(dataset[col_behav])

        # Computes the number of points of each behavioral values.
        self.__distrib = Counter(self.__val_bhv)

        # Gets the dataset characteristics.
        n_attr = len(col_contex)
        self.__n_pts = len(dataset)
        n_cls = len(self.__distrib)

        # Computes the tolerance limit and the number of quantiles.
        self.__limit = Hino.limit_estimator(n_cls, n_attr)
        self.__n_qtils = Hino.n_quantiles_estimator(self.__n_pts, n_cls)
        self.__max_p_outliers = None

    @property
    def limit(self) -> int:
        """
        The tolerance limit that will be applied (if detection has not been
        started) or that has been applied (if detection has been started).
        This value may change after detection if too much outlier has been
        detected with the previous limit.

        Return
        ------
        int
            the tolerance limit.
        """
        return self.__limit

    @property
    def n_quantiles(self) -> int:
        """
        The quantile number used for point separation.

        Return
        ------
        int
            the number of quantiles.
        """
        return self.__n_qtils

    @staticmethod
    def limit_estimator(n_cls: int, n_attr: int) -> int:
        """
        Calculates the optimal tolerance limit for outlier detection according
        to the characteristics of a dataset.
        It is calculated using the linear regression formula, the steps of
        which are discussed in the article.

        Parameters
        ----------
        n_cls: int
            The number of behavioral values (also called class) in the dataset.
        n_attr: int
            The number of contextual attributes in the dataset, i.e. the
            attributes that represent the specific characteristics of a point.

        Return
        ------
        int
            the integer value closest to the calculated optimum limit.
        """
        return round((0.0205 * log2(-1.623730 + n_cls) + 0.062579) * n_attr)

    @staticmethod
    def n_quantiles_estimator(n_pts: int, n_cls: int) -> float:
        """
        Calculates the optimal number of quantiles for outlier detection
        according to the characteristics of a dataset.
        It was suggested in the article that quantiles should be small enough
        to contain points that share similar properties, but large enough that
        they could contain at least one representative of each behavioral
        value.
        Thus, the quantile number is equal to:
        n = n_{p} / (n_{v_{B}} + 1)
        where n_{p} is the number of points and n_{v_{B}} is the number of
        behavioral values.

        Parameters
        ----------
        n_cls: int
            The number of behavioral values (also called class) in the dataset.
        n_attr: int
            The number of contextual attributes in the dataset, i.e. the
            attributes that represent the specific characteristics of a point.

        Return
        ------
        int
            the integer value closest to the calculated optimum number of
            quantiles.
        """
        return round(n_pts / (n_cls + 1))

    def set_limit(self, limit: int) -> None:
        """
        Changes the tolerance limit to be applied to the next detection.

        Parameters
        ----------
        limit: int
            The new tolerance limit.
            It must be greater than or equal to 0.

        Raises
        ------
        ValueError
            When the given tolerance limit is strictly lower than 0.
        """
        if limit < 0:
            raise ValueError("The tolerance limit must be greater than or ",
                             "equal to 0.")
        self.__limit = limit

    def set_n_quantiles(self, n_qtils: int) -> None:
        """
        Changes the number of quantiles to be applied to the next detection.

        Parameters
        ----------
        n_qtils: int
            The new number of quantiles.
            It must be greater than or equal to 2.

        Raises
        ------
        ValueError
            When the given number of quantiles is strictly lower than 2.
        """
        if n_qtils < 2:
            raise ValueError("The number of quantiles must be greater than or",
                             " equal to 0")
        self.__n_qtils = n_qtils

    def set_max_percent_outliers_detected(self, percent: float) -> None:
        """
        Sets the percentage of allowed outliers to be applied to the next
        detection.

        Parameters
        ----------
        percent: float
            The percentage of allowed outliers.
            It must be contains in the interval ]0.0; 1.0[.

        Raises
        ------
        ValueError
            When the given percentage is not in the interval ]0.0; 1.0[.
        """
        if percent < 0.0 or percent > 1.0:
            raise ValueError("The percentage of allowed outliers must be in ",
                             "the interval ]0.0; 1.0[.")
        self.__max_p_outliers = percent

    def __limit_review(self, isolation: List[int], min_p: float):
        """
        Modify and re-evaluate the tolerance limit according to the isolation
        scores of each point, so that the maximum percentage of outlier allowed
        is respected.

        Parameters
        ----------
        isolation: List[int]
            Number of times a point (at the same index) is isolated.
        min_p: float
            Maximum percentage of outlier allowed.
        """
        count = Counter(isolation)
        count = dict(sorted(count.items(), reverse=True))

        n_outliers = 0
        p_pts = 0.0
        i = -1
        keys = list(count.keys())
        while (p_pts < min_p) and (i < len(keys)):
            i += 1
            n_outliers += count[keys[i]]
            p_pts = n_outliers / self.__n_pts

        limit = keys[i]
        if i == len(keys) - 1:
            limit += 1

        return limit

    def __update_isolation(self, isolation: List[int],
                           absent_bhv: List[int | str], qtil: List[int]
                           ) -> None:
        """
        Private function that update the isolation's list according to the
        behavioral values absent in the direct adjacents quantile.

        Parameters
        ----------
        isolation: List[int]
            Number of times a point (at the same index) is isolated.
        absent_bhv: List[int | str]
            Behavioral values present in the current quantile but absent in the
            two direct adjacents ones.
        qtil: List[int]
            Indices of point in the current quantile.
        """
        if len(absent_bhv) > 0:
            for i in qtil:
                if self.__val_bhv[i] in absent_bhv:
                    isolation[i] += 1

    def __n_cdts_failed(self) -> List[int]:
        """
        Private function that counts the number of times a point is isolated.
        As a reminder, a point is isolated when no point present in the two
        adjacent direct quantiles has the same behavioral value as it.
        Unless all the points of this behavioral value are present in this
        quantile, in which case they cannot be isolated.

        Return
        ------
        List[int]
            the number of times a point (at the same index) is isolated.
        """
        # Splits the data set into several quantiles and count the number of
        # occurrences of behavioral values in each of them.
        ctx_qtils = Quantile.quantiles(self.__val_ctx, self.__n_qtils)
        i_pts_qtils = Quantile.points_per_quantiles(self.__val_ctx, ctx_qtils)
        ctx_distrib = Quantile.quantiles_distribution(self.__val_bhv,
                                                      i_pts_qtils)

        # Initializes the list who stores the count of number of time a point
        # doesn't satisfy our condition, i.e. the number of contextual
        # attributes where a point is isolated.
        isolation = [0 for i in range(0, len(self.__val_bhv))]

        for i_col, qtils in enumerate(i_pts_qtils):
            distribution = ctx_distrib[i_col]

            # The first quantile only checks for the presence of behavioral
            # values in its right quantile.
            # While the last quantile only checks for the presence in its
            # left quantile.
            indices = [(0, 1), (len(distribution)-1, len(distribution)-2)]
            for r, l in indices:
                # No need to check a quantile that has no points.
                if len(qtils[r]) > 0:
                    absent_bhv = [
                        e for e in distribution[r].keys()
                        if (distribution[r][e] != self.__distrib[e]
                            and e not in distribution[l].keys())
                    ]
                    self.__update_isolation(isolation, absent_bhv, qtils[r])

            # For all other quantiles, check for the presence of behavioral
            # values in the two adjacent quantiles.
            for i in range(1, len(distribution)-1):
                # No need to check a quantile that has no points.
                if len(qtils[i]) > 0:
                    absent_bhv = [
                        e for e in distribution[i].keys()
                        if (distribution[i][e] != self.__distrib[e]
                            and e not in distribution[i-1].keys()
                            and e not in distribution[i+1].keys())
                    ]
                    self.__update_isolation(isolation, absent_bhv, qtils[i])

        return isolation

    def __is_outliers(self, n_cdt_pts: List[int], limit: int) -> List[int]:
        """
        Private function that indicated if a point is an outliers according
        the tolerance limit choosen.

        Parameters
        ----------
        n_cdt_pts: List[int]
            The number of times a point (at this index) is isolated.
        limit: int
            The tolerance limit.

        Return
        ------
        List[int]
            Indicates if the point (at this index) is an outliers (1) or not
            (0).
        """
        return [1 if occur > limit else 0 for occur in n_cdt_pts]

    def fit(self) -> List[int]:
        """
        Runs outlier detection according to the number of quantiles and
        tolerance limit selected or estimated.

        Return
        ------
            Indicates if the point (at this index) is an outliers (1) or not
            (0).
        """
        # Computes the isolation score of each point and if they are an
        # outliers or not.
        n_cdt_pts = self.__n_cdts_failed()
        result = self.__is_outliers(n_cdt_pts, self.__limit)

        # If a maximum percentage of outliers was choosen and if the actual
        # percentage is higher, then re-estimate the tolerance limit and
        # re-compute the outliers detection.
        if self.__max_p_outliers is not None:
            p_outliers = sum(result) / len(result)
            if p_outliers > self.__max_p_outliers:
                del result
                self.__limit = self.__limit_review(n_cdt_pts,
                                                   self.__max_p_outliers)
                result = self.__is_outliers(n_cdt_pts, self.__limit)

        return result
