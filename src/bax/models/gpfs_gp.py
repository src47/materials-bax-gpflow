"""
Code for Gaussian processes using GPflow and GPflowSampling.
"""

from argparse import Namespace
import copy
import numpy as np
import tensorflow as tf
from gpflow import kernels
from gpflow import optimizers
from gpflow.utilities import print_summary

from gpflow.config import default_float as floatx
from gpflow import set_trainable
from .simple_gp import SimpleGp
from .gpfs.models import PathwiseGPR
from .gp.gp_utils import kern_exp_quad, kern_exp_quad_ard
from ..util.base import Base
from ..util.misc_util import dict_to_namespace, suppress_stdout_stderr
from copy import deepcopy
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow_probability as tfp
import gpflow


class GpfsGp(SimpleGp):
    """
    GP model using GPFlowSampling.
    """

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        super().set_params(params)
        params = dict_to_namespace(params)

        # Set self.params
        self.params.name = getattr(params, "name", "GpfsGp")
        self.params.n_bases = getattr(params, "n_bases", 1000)
        self.params.n_dimx = getattr(params, "n_dimx", 1)
        self.params.use_ard = getattr(params, "use_ard", False)
        self.params.ls = getattr(params, "ls", None)
        self.params.fixed_noise = getattr(params, "fixed_noise", False)

        if self.params.ls is None:
            if self.params.use_ard:
                self.params.ls = [3.17] * self.params.n_dimx
            else:
                self.params.ls = 3.17

        self.set_kernel(params)

    def set_kernel(self, params):
        """Set GPflow kernel."""
        self.params.kernel_str = getattr(params, "kernel_str", "rbf")

        ls = self.params.ls
        kernvar = self.params.alpha**2
        if self.params.kernel_str == "rbf":
            if self.params.use_ard:
                gpf_kernel = kernels.SquaredExponential(variance=kernvar, lengthscales=ls)
                kernel = getattr(params, "kernel", kern_exp_quad_ard)
            else:
                gpf_kernel = kernels.SquaredExponential(variance=kernvar, lengthscales=ls)
                kernel = getattr(params, "kernel", kern_exp_quad)
        elif self.params.kernel_str == "matern52":
            gpf_kernel = kernels.Matern52(variance=kernvar, lengthscales=ls)
            raise Exception("Matern 52 kernel is not yet supported.")
        elif self.params.kernel_str == "matern32":
            gpf_kernel = kernels.Matern32(variance=kernvar, lengthscales=ls)
            raise Exception("Matern 32 kernel is not yet supported.")

        self.params.gpf_kernel = gpf_kernel
        self.params.kernel = kernel

    def set_data(self, data):
        """Set self.data."""
        super().set_data(data)
        self.tf_data = Namespace()
        self.tf_data.x = tf.convert_to_tensor(np.array(self.data.x))
        self.tf_data.y = tf.convert_to_tensor(np.array(self.data.y).reshape(-1, 1))
        self.set_model()

    def set_model(self):
        """Set GPFlowSampling as self.model."""
        self.params.model = PathwiseGPR(
            data=(self.tf_data.x, self.tf_data.y),
            kernel=self.params.gpf_kernel,
            noise_variance=self.params.sigma**2,
        )
        if self.params.fixed_noise:
            set_trainable(self.params.model.likelihood.variance, False)

    def initialize_function_sample_list(self, n_fsamp=1):
        """Initialize a list of n_fsamp function samples."""
        n_bases = self.params.n_bases
        paths = self.params.model.generate_paths(num_samples=n_fsamp, num_bases=n_bases)
        _ = self.params.model.set_paths(paths)

        Xinit = tf.random.uniform([n_fsamp, self.params.n_dimx], minval=0.0, maxval=0.1, dtype=floatx())
        Xvars = tf.Variable(Xinit)
        self.fsl_xvars = Xvars
        self.n_fsamp = n_fsamp

    @tf.function
    def call_fsl_on_xvars(self, model, xvars, sample_axis=0):
        """Call fsl on fsl_xvars."""
        fvals = model.predict_f_samples(Xnew=xvars, sample_axis=sample_axis)
        return fvals

    def call_function_sample_list(self, x_list):
        """Call a set of posterior function samples on respective x in x_list."""

        # Replace Nones in x_list with first non-None value
        x_list = self.replace_x_list_none(x_list)

        # Set fsl_xvars as x_list, call fsl, return y_list
        self.fsl_xvars.assign(x_list)

        y_tf = self.call_fsl_on_xvars(self.params.model, self.fsl_xvars)
        y_list = list(y_tf.numpy().reshape(-1))
        return y_list

    def call_function_sample_list_mean(self, x):
        """
        Call a set of posterior function samples on an input x and return mean of
        outputs.
        """

        # Construct x_dupe_list
        x_dupe_list = [x for _ in range(self.n_fsamp)]

        # Set fsl_xvars as x_dupe_list, call fsl, return y_list
        self.fsl_xvars.assign(x_dupe_list)
        y_tf = self.call_fsl_on_xvars(self.params.model, self.fsl_xvars)
        y_mean = y_tf.numpy().reshape(-1).mean()
        return y_mean

    def replace_x_list_none(self, x_list):
        """Replace any Nones in x_list with first non-None value and return x_list."""

        # Set new_val as first non-None element of x_list
        new_val = next(x for x in x_list if x is not None)

        # Replace all Nones in x_list with new_val
        x_list_new = [new_val if x is None else x for x in x_list]

        return x_list_new

    def fit_hypers(self, return_hyper_dict: bool = False):
        """Fit hyperparameters."""
        opt = optimizers.Scipy()
        opt_config = dict(maxiter=getattr(self.params, "opt_max_iter", 1e5))
        print_hyps = getattr(self.params, "print_fit_hypers", False)
        # Fit hyperparameters
        if print_hyps:
            print("GPflow: start hyperparameter fitting.")
        opt_log = opt.minimize(
            self.params.model.training_loss,
            self.params.model.trainable_variables,
            options=opt_config,
        )
        if print_hyps:
            print("GPflow: end hyperparameter fitting.")
            print_summary(self.params.model)


class MultiGpfsGp(Base):
    """
    Simple multi-output GP model using GPFlowSampling. To do this, this class duplicates
    the model in GpfsGp multiple times (and uses same kernel and other parameters in
    each duplication).
    """

    def __init__(self, params=None, data=None, verbose=True):
        super().__init__(params, verbose)
        self.set_data(data)
        self.set_gpfsgp_list()

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, "name", "MultiGpfsGp")
        self.params.n_dimy = getattr(params, "n_dimy", 1)
        self.params.gp_params = getattr(params, "gp_params", {})

    def set_data(self, data):
        """Set self.data."""
        if data is None:
            self.data = Namespace(x=[], y=[])
        else:
            data = dict_to_namespace(data)
            self.data = copy.deepcopy(data)

    def set_gpfsgp_list(self):
        """Set self.gpfsgp_list by instantiating a list of GpfsGp objects."""
        data_list = self.get_data_list(self.data)
        gp_params_list = self.get_gp_params_list()

        # Each GpfsGp verbose set to same as self.params.verbose
        verb = self.params.verbose
        self.gpfsgp_list = [GpfsGp(gpp, dat, verb) for gpp, dat in zip(gp_params_list, data_list)]

    def initialize_function_sample_list(self, n_samp=1):
        """
        Initialize a list of n_samp function samples, for each GP in self.gpfsgp_list.
        """
        for gpfsgp in self.gpfsgp_list:
            gpfsgp.initialize_function_sample_list(n_samp)

    def call_function_sample_list(self, x_list):
        """
        Call a set of posterior function samples on respective x in x_list, for each GP
        in self.gpfsgp_list.
        """
        y_list_list = [gpfsgp.call_function_sample_list(x_list) for gpfsgp in self.gpfsgp_list]

        # y_list is a list, where each element is a list representing a multidim y
        y_list = [list(x) for x in zip(*y_list_list)]
        return y_list

    def call_function_sample_list_mean(self, x):
        """
        Call a set of posterior function samples on an input x and return mean of
        outputs, for each GP in self.gpfsgp_list.
        """

        # y_vec is a list of outputs for a single x (one output per gpfsgp)
        y_vec = [gpfsgp.call_function_sample_list_mean(x) for gpfsgp in self.gpfsgp_list]
        return y_vec

    def get_post_mu_cov(self, x_list, full_cov=False):
        """See SimpleGp. Returns a list of mu, and a list of cov/std."""
        mu_list, cov_list = [], []
        for gpfsgp in self.gpfsgp_list:
            # Call usual 1d gpfsgp gp_post_wrapper
            mu, cov = gpfsgp.get_post_mu_cov(x_list, full_cov)
            mu_list.append(mu)
            cov_list.append(cov)

        return mu_list, cov_list

    def gp_post_wrapper(self, x_list, data, full_cov=True):
        """See SimpleGp. Returns a list of mu, and a list of cov/std."""

        data_list = self.get_data_list(data)
        mu_list = []
        cov_list = []

        for gpfsgp, data_single in zip(self.gpfsgp_list, data_list):
            # Call usual 1d gpfsgp gp_post_wrapper
            mu, cov = gpfsgp.gp_post_wrapper(x_list, data_single, full_cov)
            mu_list.append(mu)
            cov_list.append(cov)

        return mu_list, cov_list

    def get_data_list(self, data):
        """
        Return list of Namespaces, where each is a version of data containing only one
        of the dimensions of data.y (and the full data.x).
        """

        data_list = []
        for j in range(self.params.n_dimy):
            data_list.append(Namespace(x=data.x, y=[yi[j] for yi in data.y]))

        return data_list

    def get_gp_params_list(self):
        """
        Return list of gp_params dicts (same length as self.data_list), by parsing
        self.params.gp_params.
        """
        gp_params_list = [copy.deepcopy(self.params.gp_params) for _ in range(self.params.n_dimy)]

        hyps = ["ls", "alpha", "sigma"]
        for hyp in hyps:
            if not isinstance(self.params.gp_params.get(hyp, 1), (float, int)):
                # If hyp exists in dict, and is not (float, int), assume is list of hyp
                for idx, gpp in enumerate(gp_params_list):
                    gpp[hyp] = self.params.gp_params[hyp][idx]

        return gp_params_list


class BatchGpfsGp(GpfsGp):
    """
    GPFlowSampling GP model tailored to batch algorithms with BAX.
    """

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        super().set_params(params)
        params = dict_to_namespace(params)

        # Set self.params
        self.params.name = getattr(params, "name", "BatchGpfsGp")

    def initialize_function_sample_list(self, n_fsamp=1):
        """Initialize a list of n_fsamp function samples."""
        n_bases = self.params.n_bases
        paths = self.params.model.generate_paths(num_samples=n_fsamp, num_bases=n_bases)
        _ = self.params.model.set_paths(paths)
        self.n_fsamp = n_fsamp

    def initialize_fsl_xvars(self, n_batch):
        """
        Initialize set.fsl_xvars, a tf.Variable of correct size, given batch size
        n_batch.
        """
        Xinit = tf.zeros([self.n_fsamp, n_batch, self.params.n_dimx], dtype=floatx())
        Xvars = tf.Variable(Xinit)
        self.fsl_xvars = Xvars

    def call_function_sample_list(self, x_batch_list):
        """
        Call a set of posterior function samples on respective x_batch (a list of
        inputs) in x_batch_list.
        """
        # Replace empty x_batch and convert all x_batch to max batch size
        x_batch_list_new, max_n_batch = self.reformat_x_batch_list(x_batch_list)
        # Only re-initialize fsl_xvars if max_n_batch is larger than self.max_n_batch
        # Viraj: I observerd that this doesn't work -- it still crashes if you pass in a batch that is smaller than
        # self.max_n_batch. So I'm changing this to !=
        if max_n_batch != getattr(self, "max_n_batch", 0):
            self.max_n_batch = max_n_batch
            self.initialize_fsl_xvars(max_n_batch)
        # Set fsl_xvars as x_batch_list_new
        self.fsl_xvars.assign(x_batch_list_new)

        # Call fsl on fsl_xvars, return y_list
        y_tf = self.call_fsl_on_xvars(self.params.model, self.fsl_xvars)

        # Return list of y_batch lists, each cropped to same size as original x_batch
        y_batch_list = []
        for yarr, x_batch in zip(y_tf.numpy(), x_batch_list):
            y_batch = list(yarr.reshape(-1))[: len(x_batch)]
            y_batch_list.append(y_batch)

        return y_batch_list

    def reformat_x_batch_list(self, x_batch_list):
        """Make all batches the same size and replace all empty lists."""

        # Find first non-empty list and use first entry as dummy value
        dum_val = next(x_batch for x_batch in x_batch_list if len(x_batch) > 0)[0]
        max_n_batch = max(len(x_batch) for x_batch in x_batch_list)

        # duplicate and reformat each x_batch in x_batch_list, add to x_batch_list_new
        x_batch_list_new = []
        for x_batch in x_batch_list:
            x_batch_dup = [*x_batch]
            x_batch_dup.extend([dum_val] * (max_n_batch - len(x_batch_dup)))
            x_batch_list_new.append(x_batch_dup)

        return x_batch_list_new, max_n_batch

    def fit_hypers(self, return_hyper_dict: bool = False):
        """Fit hyperparameters."""
        opt = optimizers.Scipy()
        opt_config = dict(maxiter=getattr(self.params, "opt_max_iter", 1e5))
        print_hyps = getattr(self.params, "print_fit_hypers", False)
        # print_hyps = True
        # Fit hyperparameters
        if print_hyps:
            print("GPflow: start hyperparameter fitting.")
            loss_start = self.params.model.training_loss()

        self.params.model.kernel.lengthscales = gpflow.Parameter(
            self.params.n_dimx
            * [
                0.1,
            ],
            transform=tfp.bijectors.SoftClip(
                gpflow.utilities.to_default_float(0.01),
                gpflow.utilities.to_default_float(2.0),
            ),
        )

        self.params.model.kernel.variance = gpflow.Parameter(
            1.0,
            transform=tfp.bijectors.SoftClip(
                gpflow.utilities.to_default_float(0.01),
                gpflow.utilities.to_default_float(4),
            ),
        )

        opt_log = opt.minimize(
            self.params.model.training_loss,
            self.params.model.trainable_variables,
            options=opt_config,
        )

        if print_hyps:
            print("GPflow: end hyperparameter fitting.")
            loss_end = self.params.model.training_loss()
            print(loss_start, loss_end)
            print_summary(self.params.model)

        ## Update model hypers:
        if self.params.use_ard:
            self.params.ls = list(self.params.model.kernel.lengthscales.numpy())
        else:
            self.params.ls = float(self.params.model.kernel.lengthscales.numpy())

        self.params.alpha = float(np.sqrt(self.params.model.kernel.variance.numpy()))
        self.params.sigma = float(np.sqrt(self.params.model.likelihood.variance.numpy()))

        if return_hyper_dict:
            hpdict = dict(
                n_dimx=self.params.n_dimx,
                use_ard=self.params.use_ard,
                fixed_noise=self.params.fixed_noise,
                ls=self.params.ls,
                alpha=self.params.alpha,
                sigma=self.params.sigma,
            )

            return hpdict


class BatchMultiGpfsGp(MultiGpfsGp):
    """
    Batch version of MultiGpfsGp model, which is tailored to batch algorithms with BAX.
    To do this, this class duplicates the model in BatchGpfsGp multiple times.
    """

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, "name", "MultiBatchGpfsGp")

    def set_gpfsgp_list(self):
        """Set self.gpfsgp_list by instantiating a list of BatchGpfsGp objects."""
        data_list = self.get_data_list(self.data)
        gp_params_list = self.get_gp_params_list()
        # Each BatchGpfsGp verbose set to same as self.params.verbose
        verb = self.params.verbose
        self.gpfsgp_list = [BatchGpfsGp(gpp, dat, verb) for gpp, dat in zip(gp_params_list, data_list)]

    def call_function_sample_list(self, x_batch_list):
        """
        Call a set of posterior function samples on respective x in x_list, for each GP
        in self.gpfsgp_list.
        """
        y_batch_list_list = [gpfsgp.call_function_sample_list(x_batch_list) for gpfsgp in self.gpfsgp_list]

        # We define y_batch_multi_list to be a list, where each element is: a list of
        # multi-output-y (one per n_batch)
        y_batch_multi_list = [list(zip(*ybl)) for ybl in zip(*y_batch_list_list)]  # ugly

        # Convert from list of lists of tuples to all lists
        y_batch_multi_list = [[list(tup) for tup in li] for li in y_batch_multi_list]

        return y_batch_multi_list

    def call_function_sample_list_mean(self, x):
        """
        Call a set of posterior function samples on an input x and return mean of
        outputs, for each GP in self.gpfsgp_list.
        """
        # TODO: possibly implement for BatchMultiGpfsGp for sample approximation of
        # posterior mean
        pass

    def fit_hypers(self, return_hypers_dict: bool = False):
        ct = 0
        hp_dict = None
        for model in self.gpfsgp_list:
            if return_hypers_dict:
                tmp_dict = model.fit_hypers(return_hypers_dict)
                if ct == 0:
                    hp_dict = tmp_dict.copy()
                    hp_dict.update(
                        {
                            "ls": [tmp_dict["ls"]],
                            "alpha": [tmp_dict["alpha"]],
                            "sigma": [tmp_dict["sigma"]],
                        }
                    )
                else:
                    hp_dict.update(
                        {
                            "ls": hp_dict["ls"] + [tmp_dict["ls"]],
                            "alpha": hp_dict["alpha"] + [tmp_dict["alpha"]],
                            "sigma": hp_dict["sigma"] + [tmp_dict["sigma"]],
                        }
                    )

            else:
                model.fit_hypers(return_hypers_dict)
            ct += 1
        if return_hypers_dict:
            return hp_dict


# def find_best_hypers(data_all, params_model, fraction=0.1, seed: int = 0):
#     breakpoint()
#     local_params_model = deepcopy(params_model)

#     ndpts = np.array(data_all.x).shape[0]
#     x = np.array(data_all.x)
#     y = np.array(data_all.y)
#     nobj = y.shape[1]
#     ndtps_hyper = int(ndpts * fraction)
#     folds = ndpts // ndtps_hyper
#     kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

#     rmses = np.ones(nobj) * 1e9
#     best_params = None
#     # We flip train and val so the training set is smaller
#     for i, (train_index, val_index) in enumerate(kf.split(x)):
#         data_for_fixed_hypers = Namespace()
#         data_for_fixed_hypers.x = [tuple(xi) for xi in x[val_index]]
#         data_for_fixed_hypers.y = [tuple(yi) for yi in y[val_index]]

#         fix_hypers_model = BatchMultiGpfsGp(params_model, data_for_fixed_hypers, False)
#         hp_dict = fix_hypers_model.fit_hypers(True)

#         local_params_model.update({"gp_params": hp_dict})

#         fix_hypers_model = BatchMultiGpfsGp(local_params_model, data_for_fixed_hypers, False)
#         mu_pred, _ = fix_hypers_model.get_post_mu_cov(x[train_index], False)
#         posterior_mean = np.array(mu_pred).T

#         tmp_rmses = np.ones(nobj) * 1e9
#         for iobj in range(0, nobj):
#             tmp_rmses[iobj] = np.sqrt(mean_squared_error(y[train_index, iobj], posterior_mean[:, iobj]))
#         if np.mean(rmses) < np.mean(tmp_rmses):
#             pass
#         else:
#             for iobj in range(0, nobj):
#                 rmses[iobj] = tmp_rmses[iobj]
#             best_params = deepcopy(local_params_model)

#     return best_params, rmses


def find_best_hypers(data_all, params_model, fraction=0.2, seed: int = 0):
    local_params_model = deepcopy(params_model)
    ndpts = np.array(data_all.x).shape[0]
    x = np.array(data_all.x)
    y = np.array(data_all.y)
    nobj = y.shape[1]
    ndtps_hyper = int(ndpts * fraction)
    folds = ndpts // ndtps_hyper
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    rmses = np.ones(nobj) * 1e9
    best_params = None
    # We flip train and val so the training set is smaller
    for i, (train_index, val_index) in enumerate(kf.split(x)):
        data_for_fixed_hypers = Namespace()
        data_for_fixed_hypers.x = [tuple(xi) for xi in x[train_index]]
        data_for_fixed_hypers.y = [tuple(yi) for yi in y[train_index]]

        fix_hypers_model = BatchMultiGpfsGp(params_model, data_for_fixed_hypers, False)
        hp_dict = fix_hypers_model.fit_hypers(True)

        local_params_model.update({"gp_params": hp_dict})

        fix_hypers_model = BatchMultiGpfsGp(local_params_model, data_for_fixed_hypers, False)
        mu_pred, _ = fix_hypers_model.get_post_mu_cov(x[val_index], False)
        posterior_mean = np.array(mu_pred).T

        tmp_rmses = np.ones(nobj) * 1e9
        for iobj in range(0, nobj):
            tmp_rmses[iobj] = np.sqrt(mean_squared_error(y[val_index, iobj], posterior_mean[:, iobj]))
        if np.mean(rmses) < np.mean(tmp_rmses):
            pass
        else:
            for iobj in range(0, nobj):
                rmses[iobj] = tmp_rmses[iobj]
            best_params = deepcopy(local_params_model)
    return best_params, rmses
