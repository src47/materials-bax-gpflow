import numpy as np
from numpy.typing import NDArray

from subspace_algorithm.bax_subspace_algorithm_base import BaxSubspaceAlgorithm
from subspace_algorithm.subspace_algorithm_utils.helper_subspace_functions import (
    discontinous_library_1d,
    multi_level_region_union_Nd,
    multi_level_region_intersection_Nd,
    sublevelset,
    obtain_discrete_pareto_optima,
    convert_y_for_optimization,
    wish_list,
)

FloatArray = NDArray[np.float_]
IntArray = NDArray[np.int_]

"""

Class which implements various subspace algorithms. Inherits from BaxSubspaceAlgorithm. Here a user provided 
parameter dictionary is needed.

"""


class WishList(BaxSubspaceAlgorithm):
    """
    Wish list algorithm
    """

    def identify_subspace(self, x: FloatArray = None, y: FloatArray = None):
        list_of_wishes = self.params.subregion_algo_params["list_of_wishes"]
        y = np.array(y)

        # assert y.ndim == 1

        desired_indices = wish_list(y, list_of_wishes)
        return desired_indices


class FunctionEstimation(BaxSubspaceAlgorithm):
    """
    Subspace algorithm to perform full function estimation.
    In this algorithm, all points in the X domain are trivially returned.
    Using infoBAX with this algorithm gives similar sampling patterns to
    uncertainty sampling.
    """

    def identify_subspace(self, x: FloatArray = None, y: FloatArray = None):
        desired_indices = np.arange(0, len(y))  # all points in domain are desired
        return np.array(desired_indices, dtype=int)


class GlobalOptimization1D(BaxSubspaceAlgorithm):
    """
    Subspace algorithm to estimate the global minimum/maximum of a function.
    In this algorithm, the desired point is index corresponding to the smallest/largest value in y.
    Here, y is a 1D array. Using infoBAX with this algorithm gives similar sampling patterns to classical
    BO algorithms such as EIG. Note, in this implementation, ties are handled by taking the first occurence.

    subregion_algo_params is a dictionary that is inherited from BaxSubspaceAlgorithm.

    For this algorithm, the following parameters are passed:

    - min: boolean for whether to perform minimization or maximization.
    """

    def identify_subspace(self, x: FloatArray = None, y: FloatArray = None):
        y = np.array(y)

        assert (
            y.ndim == 1
        ), "GlobalOptimization1D only takes in one property. Please ensure that y is a 1D array."

        algo_params = self.params.subregion_algo_params
        desired_axis = int(algo_params["axis"])
        is_min = int(algo_params["min"])

        if is_min:
            desired_indices = [
                int(np.argmin(y[:, desired_axis]))
            ]  # Only the first occurrence is returned.
        else:
            desired_indices = [int(np.argmax(y[:, desired_axis]))]

        assert (
            len(desired_indices) == 1
        ), "GlobalOptimization1D.identify_subspace should only return one index"

        return np.array(desired_indices, dtype=int)


class RobustParetoFront(BaxSubspaceAlgorithm):
    """
    Subspace algorithm to estimate the pareto front of multiple properties with a tolerance factor.
    Using infoBAX with this algorithm gives similar sampling patterns to MOBO algorithms such as EHVI.

    subregion_algo_params is a dictionary that is inherited from BaxSubspaceAlgorithm.

    For this algorithm, the following parameters are passed:

    - tolerance_list: Tolerance for both properties. The algorithm will first search for y's on the pareto front and
                      then add in y's which correspond to y[i] + tolerance_list[i].
    - max_or_min_list: boolean vector for whether the property should be maximized (1) or minimized (0)
    """

    def identify_subspace(self, x: FloatArray = None, y: FloatArray = None):
        subregion_algo_params = self.params.subregion_algo_params  # unused here
        tolerance_list = subregion_algo_params["tolerance_list"]
        max_or_min_list = subregion_algo_params["max_or_min_list"]

        y = convert_y_for_optimization(y, max_or_min_list)
        error_bars = tolerance_list * np.ones(np.array(y).shape)
        desired_indices = obtain_discrete_pareto_optima(
            np.array(x), np.array(y), error_bars=error_bars
        )
        return np.array(desired_indices, dtype=int)


class PercentileSet(BaxSubspaceAlgorithm):

    """
    PercentileSet to find the set of points which are in the top specified percentile of the given points.

    - percentile: percentile threshold.

    """

    def identify_subspace(self, x: FloatArray = None, y: FloatArray = None):
        percentile = self.params.subregion_algo_params["percentile"]
        y = np.array(y)

        # assert y.ndim == 1

        top_percentile_value = np.percentile(y, percentile)
        desired_indices = list(set(np.where(y <= top_percentile_value)[0]))
        return desired_indices


class LevelSet(BaxSubspaceAlgorithm):

    """
    LevelSet to find the set of points which are in the top specified percentile of the given points.

    - LevelSet: LevelSet threshold.

    """

    def identify_subspace(self, x: FloatArray = None, y: FloatArray = None):
        threshold = self.params.subregion_algo_params["threshold"]
        y = np.array(y)

        # assert y.ndim == 1

        desired_indices = list(set(np.where(y <= threshold)[0]))
        return desired_indices


class Percentile2D(BaxSubspaceAlgorithm):

    """
    Percentile2D is a generalized percentile set algorithm for 2 properties. It can either take the union, intersection
    or condition intersection - union of the properties.

    - percentile_list: list of property thresholds.
    - combination_type: union, intersection or conditional
    - max_or_min_list: boolean vector for whether the property should be maximized (1) or minimized (0)

    """

    def identify_subspace(self, x: FloatArray = None, y: FloatArray = None):
        assert (
            np.array(y).ndim == 2
        ), "Percentile2D only takes in two properties. Future release will generalize this N-dimensions."

        subregion_algo_params = self.params.subregion_algo_params

        percentile_list = subregion_algo_params["percentile_list"]
        combination_type = subregion_algo_params["combination_type"]
        max_or_min_list = subregion_algo_params["max_or_min_list"]

        y = convert_y_for_optimization(y, max_or_min_list)

        y1 = np.array(y)[:, 0]
        y2 = np.array(y)[:, 1]

        top_percentile_value_1 = np.percentile(y1, percentile_list[0])
        top_percentile_value_2 = np.percentile(y2, percentile_list[1])

        s1 = set(np.where(y1 >= top_percentile_value_1)[0])
        s2 = set(np.where(y2 >= top_percentile_value_2)[0])

        union_ids = list(s1.union(s2))
        intersection_ids = list(s1.intersection(s2))

        # Choose how to combine percentile sets
        if combination_type == "union":
            desired_indices = union_ids
        elif combination_type == "intersection":
            desired_indices = intersection_ids
        elif combination_type == "conditional":
            desired_indices = intersection_ids
            if intersection_ids == []:
                desired_indices = union_ids
        else:
            raise Exception(
                "Percentile2D only supports union, intersection and conditional as combination_types"
            )
        return desired_indices


class MultiRegionSetUnion(BaxSubspaceAlgorithm):

    """
    MultiRegionSetUnion gets the union of multiple level region sets. Works in arbitrary dimensions.
    - threshold_list: A list of min/max values for different properties.
    """

    def identify_subspace(self, x: FloatArray = None, y: FloatArray = None):
        subregion_algo_params = self.params.subregion_algo_params
        threshold_list = subregion_algo_params["threshold_list"]
        desired_indices = multi_level_region_union_Nd(y, threshold_list)

        return desired_indices


class MultiRegionSetIntersection(BaxSubspaceAlgorithm):

    """
    MultiRegionSetIntersection gets the intersection of multiple level region sets. Works in arbitrary dimensions.
    - threshold_list: A list of min/max values for different properties.
    """

    def identify_subspace(self, x: FloatArray = None, y: FloatArray = None):
        subregion_algo_params = self.params.subregion_algo_params
        threshold_list = subregion_algo_params["threshold_list"]
        desired_indices = multi_level_region_intersection_Nd(y, threshold_list)

        return desired_indices


class ConditionalMultiRegionSetIntersection(BaxSubspaceAlgorithm):

    """
    MultiRegionSetIntersection gets the intersection of multiple level region sets. Works in arbitrary dimensions.
    - threshold_list: A list of min/max values for different properties.
    """

    def identify_subspace(self, x: FloatArray = None, y: FloatArray = None):
        subregion_algo_params = self.params.subregion_algo_params
        threshold_list_1 = subregion_algo_params["threshold_list_1"]
        desired_indices = multi_level_region_intersection_Nd(y, threshold_list_1)

        if desired_indices == []:
            threshold_list_2 = subregion_algo_params["threshold_list_2"]
            desired_indices = multi_level_region_intersection_Nd(y, threshold_list_2)

        return desired_indices


class LevelSetDisconnected(BaxSubspaceAlgorithm):

    """
    LevelSetDisconnected gets a level set in property 1 and a list of values with epsilon in property 2.
    - threshold_1: level set threshold
    - value_list: list of values desired in property 2
    - epsilon: tolerance on value_list
    """

    def identify_subspace(self, x: FloatArray = None, y: FloatArray = None):
        # Method to be overwritten

        subregion_algo_params = self.params.subregion_algo_params

        y1 = np.array(y)[:, 0]
        y2 = np.array(y)[:, 1]

        threshold_1 = subregion_algo_params["threshold_1"]
        value_list = subregion_algo_params["value_list"]
        epsilon = subregion_algo_params["epsilon"]

        # intersection of level set and disconnected list
        intersect_id = list(
            set(sublevelset(y2, threshold_1)).intersection(
                set(discontinous_library_1d(y1, value_list, eps_vals=epsilon))
            )
        )
        desired_indices = intersect_id
        return desired_indices


# Helper function so that subregion algorithm can be read by the config file
def algorithm_type(name, algo_params={}):
    if name == "LevelSetDisconnected":
        return LevelSetDisconnected(params=algo_params)
    elif name == "RobustParetoFront":
        return RobustParetoFront(params=algo_params)
    elif name == "MultiRegionSetUnion":
        return MultiRegionSetUnion(params=algo_params)
    elif name == "MultiRegionSetIntersection":
        return MultiRegionSetIntersection(params=algo_params)
    elif name == "GlobalOptimization1D":
        return GlobalOptimization1D(params=algo_params)
    elif name == "FunctionEstimation":
        return FunctionEstimation(params=algo_params)
    elif name == "PercentileSet":
        return PercentileSet(params=algo_params)
    elif name == "LevelSet":
        return LevelSet(params=algo_params)
    elif name == "Percentile2D":
        return Percentile2D(params=algo_params)
    elif name == "ConditionalMultiRegionSetIntersection":
        return ConditionalMultiRegionSetIntersection(params=algo_params)
    elif name == "WishList":
        return WishList(params=algo_params)
    else:
        raise Exception("Subregion algorithm type not in list of available algorithms.")
