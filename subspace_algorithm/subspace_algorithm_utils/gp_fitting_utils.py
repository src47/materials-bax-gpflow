import numpy as np
from argparse import Namespace
import copy
from src.bax.util.misc_util import dict_to_namespace
from src.bax.models.gpflow_gp import GpflowGp


class GpHyperFitting:
    def __init__(
        self,
        max_iter_for_hyper_fitting,
        freq_hyper_fitting,
        dim_x,
        dim_y,
        fix_gp_hypers=False,
        current_gp_hypers_dict={},
    ):
        """
        Class Object for GP Hyper Parameter Fitting

        Args:
            max_iter_for_hyper_fitting (int): Always Fit Hypers till this iter number
            freq_hyper_fitting (int): Fit hypers after max_iter_for_hyper_fitting
                                         at this frequency of iterations
            dim_x (int): Dimension of features (X)
            dim_y (int): Dimension of Objectives (y)
            fix_gp_hypers (bool, optional): Wheteher or not to Fix Hypers. Defaults to False.
            current_gp_hypers_dict (dict, optional): Previous iteration hypers
                                                    Used in case of failure. Defaults to {}.

        Raises:
            Exception: For using fixed hypers, without specifying them
        """
        self.max_iter_for_hyper_fitting = max_iter_for_hyper_fitting
        self.freq_hyper_fitting = freq_hyper_fitting
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.fix_gp_hypers = fix_gp_hypers

        if current_gp_hypers_dict == {}:
            if self.fix_gp_hypers:
                raise Exception("Cannot fix gp hypers without providing the hypers dictionary")

            self.current_gp_hypers_dict = {
                "n_dimy": self.dim_y,
                "verbose": False,
                "gp_params": {
                    "ls": self.dim_y * [self.dim_x * [1.0]],
                    "alpha": self.dim_y * [1.0],
                    "sigma": self.dim_y * [0.01],
                    "fixed_noise": True,
                    "use_ard": True,
                    "n_dimx": self.dim_x,
                },
            }
        else:
            self.current_gp_hypers_dict = current_gp_hypers_dict

    def get_gp_hypers_dynamically(self, data, iteration):

        if self.fix_gp_hypers:
            return self.current_gp_hypers_dict
        elif self.is_iteration_gp_hyper_fit(iteration):
            return self.fit_gp_hypers_multi_objective(data)
        else:
            return self.current_gp_hypers_dict

    def is_iteration_gp_hyper_fit(self, iteration_number):
        """Check if iteration number is one in which we fit hypers

        Args:
            iteration_number (int): Iteration number

        Returns:
            bool: Whether or not to fit hypers
        """
        if (iteration_number % self.freq_hyper_fitting == 0) or (iteration_number < self.max_iter_for_hyper_fitting):
            return True
        else:
            return False

    def fit_gp_hypers_multi_objective(self, data):
        """Main function of class. The fit_hyper implementation

        Args:
            data (Namespace): Stores data.x and data.y necessary for hyper fitting

        Returns:
            Dict: Hyper Parameters Dictionary # TODO: Describe this better
        """

        assert len(data.x) <= 3000, "fit_data larger than preset limit (can cause memory issues)"
        gp_params_dict = {
            "n_dimy": self.dim_y,
            "verbose": False,
            "gp_params": {
                "ls": [],
                "alpha": [],
                "sigma": [],
                "fixed_noise": True,
                "use_ard": True,
                "n_dimx": self.dim_x,
            },
        }

        try:
            for idx in range(self.dim_y):

                data_fit = Namespace(x=data.x, y=[yi[idx] for yi in data.y])
                gp_params = self.fit_gp_hypers_single_objective(data_fit)
                gp_params_dict["gp_params"]["ls"].append(gp_params["ls"])
                gp_params_dict["gp_params"]["alpha"].append(gp_params["alpha"])
                gp_params_dict["gp_params"]["sigma"].append(gp_params["sigma"])

            for i in range(len(gp_params_dict["gp_params"]["sigma"])):
                # Clip sigma if it falls below 0.01
                gp_params_dict["gp_params"]["sigma"][i] = np.clip(gp_params_dict["gp_params"]["sigma"][i], 0.01, 10000)

            self.current_gp_hypers_dict = copy.deepcopy(gp_params_dict)

        except Exception as e:
            gp_params_dict = self.current_gp_hypers_dict

        self.check_gp_hyperdims(gp_params_dict)

        return gp_params_dict

    def fit_gp_hypers_single_objective(self, data, sigma=None):
        """
        Return hypers fit by GPflow, using data Namespace (with fields x and y). Assumes y
        is a list of scalars (i.e. 1 dimensional output).
        """

        data = dict_to_namespace(data)
        model_params = dict()

        if sigma:
            model_params["sigma"] = sigma

        # Fit hypers with GpflowGp on data
        model = GpflowGp(params=model_params, data=data)
        model.fit_hypers()
        gp_hypers = {
            "ls": model.model.kernel.lengthscales.numpy().tolist(),
            "alpha": np.sqrt(float(model.model.kernel.variance.numpy())),
            "sigma": np.sqrt(float(model.model.likelihood.variance.numpy())),
            "n_dimx": self.dim_x,
        }
        return gp_hypers

    def check_gp_hyperdims(self, gp_params_dict):
        """Ensure dictionary specified has the right hyper parameter dims

        Args:
            gp_params_dict (Dictionary): Hyper Parameter dictionary of GP
        """
        assert np.array(gp_params_dict["gp_params"]["ls"]).shape == (
            self.dim_y,
            self.dim_x,
        ), "Should have lengthscale for each X (because of ARD) and for each property (because of multigp)"

        assert (
            len(gp_params_dict["gp_params"]["alpha"]) == self.dim_y
        ), "Number of alphas in GP should equal number of y dims"

        assert (
            len(gp_params_dict["gp_params"]["sigma"]) == self.dim_y
        ), "Number of sigmas in GP should equal number of y dims"
