import numpy as np
from scipy.stats import qmc
import pandas as pd

import random


"""
Functions for loading data from CSV and converting them to a callable function

"""


def find_best_hypers(data_all, params_model, fraction=0.1):
    local_params_model = deepcopy(params_model)

    ndpts = np.array(data_all.x).shape[0]
    x = np.array(data_all.x)
    y = np.array(data_all.y)
    nobj = y.shape[1]
    ndtps_hyper = int(ndpts * fraction)
    folds = ndpts // ndtps_hyper
    kf = KFold(n_splits=folds, shuffle=True)

    rmses = np.ones(nobj) * 1e9
    best_params = None
    # We flip train and val so the training set is smaller
    for i, (train_index, val_index) in enumerate(kf.split(x)):
        data_for_fixed_hypers = Namespace()
        data_for_fixed_hypers.x = [tuple(xi) for xi in x[val_index]]
        data_for_fixed_hypers.y = [tuple(yi) for yi in y[val_index]]

        fix_hypers_model = BatchMultiGpfsGp(params_model, data_for_fixed_hypers, False)
        hp_dict = fix_hypers_model.fit_hypers(True)

        local_params_model.update({"gp_params": hp_dict})

        fix_hypers_model = BatchMultiGpfsGp(
            local_params_model, data_for_fixed_hypers, False
        )
        mu_pred, _ = fix_hypers_model.get_post_mu_cov(x[train_index], False)
        posterior_mean = np.array(mu_pred).T

        tmp_rmses = np.ones(nobj) * 1e9
        for iobj in range(0, nobj):
            tmp_rmses[iobj] = np.sqrt(
                mean_squared_error(y[train_index, iobj], posterior_mean[:, iobj])
            )
        if np.mean(rmses) < np.mean(tmp_rmses):
            pass
        else:
            for iobj in range(0, nobj):
                rmses[iobj] = tmp_rmses[iobj]
            best_params = local_params_model

    return best_params, rmses


# convert discrete observations to function by returning the X,y corresponding to the closet X in the dataset


def function_from_data(X, y):
    def fn_true(x):
        distances = np.sqrt(np.sum((X - x) ** 2, axis=1))
        k_nearest_index = np.argmin(distances)
        return y[k_nearest_index]

    return fn_true


# Load data from CSV


def XY_from_csv(path_to_csv, columns_x, columns_y):
    # Load the data from the CSV file
    df = pd.read_csv(path_to_csv)

    # assert that there are no NaNs in the dataframe (chatgpt)
    assert df.isna().sum().sum() == 0, "Should not have NANs in the dataset"

    # assert that df does not have any repeated rows (chatgpt)
    assert df.duplicated().sum() == 0, "Dataset should not have repeated rows"

    X = np.array(df[columns_x])
    y = np.array(df[columns_y])

    assert X.shape[1] >= 2, "Input X must be at least two dimensional"
    return X, y


def sobol_sample(n_desired, l_bounds, u_bounds, n_dimensions=4):
    sampler = qmc.Sobol(d=n_dimensions, scramble=False)
    sample = sampler.random_base2(m=int(np.log2(n_desired)))
    sampled_points = qmc.scale(sample, l_bounds, u_bounds)
    return sampled_points


def set_all_seeds(seed):
    import tensorflow as tf

    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
