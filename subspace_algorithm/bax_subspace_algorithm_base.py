from argparse import Namespace

import numpy as np
from numpy.typing import NDArray

from src.bax.alg.algorithms import BatchAlgorithm
from src.bax.util.misc_util import dict_to_namespace


FloatArray = NDArray[np.float_]
IntArray = NDArray[np.int_]


class BaxSubspaceAlgorithm(BatchAlgorithm):

    """A class that creates algorithms based on bayesian algorithm execution
    to obtain desired subspace indices from a provided design space."""

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        super().set_params(params)
        params = dict_to_namespace(params)

        # General params
        self.params.name = getattr(params, "name", "Subspace_Selec")
        self.params.n_batch_steps = getattr(params, "n_batch_steps", 1)
        self.params.current_batch_step_idx = getattr(params, "current_batch_step_idx", 0)
        self.params.x_normalizer = getattr(params, "x_normalizer", None)
        self.params.y_normalizer = getattr(params, "y_normalizer", None)

        self.params.subregion_algo_params = getattr(params, "subregion_algo_params", {})

    def get_next_x_batch(self):
        """
        Given the current execution path (in self.exe_path), return a batch of next x
        points for the execution path. If the algorithm is complete, return None.
        """
        if self.params.current_batch_step_idx == self.params.n_batch_steps:
            return []

        next_x_batch = self.params.x_domain

        self.params.current_batch_step_idx += 1

        return next_x_batch

    def get_exe_path_crop(self):
        """
        Return the minimal execution path for output, i.e. cropped execution path,
        specific to this algorithm.
        """
        X = self.exe_path.x
        y = self.exe_path.y

        path, path_vals = self.get_path_and_vals(X, y)

        path = path.tolist()
        path_vals = path_vals.tolist()

        # Construct and return cropped execution path
        exe_path_crop = Namespace(x=path, y=path_vals)
        return exe_path_crop

    def get_path_and_vals(self, x, y):
        if self.params.x_normalizer is not None:
            x_unnorm = self.params.x_normalizer.inverse_transform(x)
        else:
            x_unnorm = x

        if self.params.y_normalizer is not None:
            y_unnorm = self.params.y_normalizer.inverse_transform(y)
        else:
            y_unnorm = y

        subspace_indices = self.identify_subspace(x_unnorm, y_unnorm)

        region_y = np.array(y)[subspace_indices]
        region_x = np.array(x)[subspace_indices]

        path = region_x
        path_vals = region_y

        return path, path_vals

    def get_output(self):
        """Return output based on self.exe_path."""
        return self.get_exe_path_crop()

    def identify_subspace(self, x: FloatArray = None, y: FloatArray = None):
        pass
