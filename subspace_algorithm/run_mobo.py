import json
import os
import pickle
from argparse import Namespace, ArgumentParser
from typing import List

from tqdm import tqdm
import numpy as np
import tensorflow as tf
import gpflow
from sklearn.preprocessing import MinMaxScaler
from gpflow import set_trainable
from trieste.data import Dataset
from trieste.space import DiscreteSearchSpace
from trieste.models import TrainableModelStack
from trieste.models.gpflow import GaussianProcessRegression
from trieste.acquisition.function import ExpectedHypervolumeImprovement
from trieste.acquisition.rule import EfficientGlobalOptimization

from src.bax.models.gpfs_gp import BatchMultiGpfsGp, find_best_hypers
from src.bax.util.misc_util import dict_to_namespace
from subspace_algorithm.subspace_algorithm_utils.utils import (
    XY_from_csv,
    function_from_data,
    set_all_seeds,
)
from subspace_algorithm.bax_subspace_algorithm_children import algorithm_type
from subspace_algorithm.subspace_algorithm_utils.metrics_evaluation import (
    EvaluationMetrics,
)


# SOME OF THIS CODE IS ADAPTED FROM: https://arxiv.org/pdf/2107.12809.pdf
def build_trieste_model(
    data: Dataset,
    num_output: int,
    lengthscales_list: List[List[float]] = [[1.0, 1.0], [1.0, 1.0]],
    alpha_list: List[float] = [1.0, 1.0],
    sigma_list: List[float] = [0.01, 0.01],
) -> TrainableModelStack:
    trieste_stacked_models = []
    gpflow_models = []

    for idx in range(num_output):
        single_obj_data = Dataset(data.query_points, tf.gather(data.observations, [idx], axis=1))
        kernel = gpflow.kernels.SquaredExponential(variance=alpha_list[idx] ** 2, lengthscales=lengthscales_list[idx])
        gpr = gpflow.models.gpr.GPR(
            single_obj_data.astuple(),
            kernel=kernel,
            noise_variance=sigma_list[idx] ** 2,
        )
        set_trainable(gpr.likelihood.variance, False)
        gpflow_models.append(gpr)
        trieste_stacked_models.append((GaussianProcessRegression(gpr), 1))

    return TrainableModelStack(*trieste_stacked_models), gpflow_models


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        help="Path to configuration file for experiment",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="a seed value for generating the initial point. This can be used when n_repeat = 1",
    )

    # Load configuration file and parameters
    args = parser.parse_args()
    config_path = args.path

    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    config = dict_to_namespace(config)

    x_full, y_full = XY_from_csv(config.dataset_path, config.columns_x, config.columns_y)

    fn = function_from_data(x_full, y_full)  # convert X, y discrete file to a callable function
    dim_x = len(config.columns_x)
    dim_y = len(config.columns_y)

    # Construct Full Dataset Here
    all_data = Namespace()
    all_data.x_unnorm = [tuple(xi) for xi in x_full]
    all_data.y_unnorm = [tuple(yi) for yi in y_full]

    x_scaler = MinMaxScaler((0, 1))
    y_scaler = MinMaxScaler((-1, 1))

    x_scaler.fit(np.vstack((np.array(config.min_values_x), np.array(config.max_values_x))))
    y_scaler.fit(np.vstack((np.array(config.min_values_y), np.array(config.max_values_y))))

    x_full_norm = x_scaler.transform(x_full) if x_scaler is not None else x_full
    y_full_norm = y_scaler.transform(y_full) if y_scaler is not None else y_full

    all_data.x = [tuple(xi) for xi in x_full_norm]
    all_data.y = [tuple(yi) for yi in y_full_norm]

    # Hyper Fitting Consistency
    tmp_model = BatchMultiGpfsGp
    params_model = {
        "n_dimy": dim_y,
        "gp_params": dict(
            n_dimx=dim_x,
            use_ard=True,
            fixed_noise=True,
            ls=[[0.1] * dim_x] * dim_y,
            alpha=[1] * dim_y,
            sigma=np.max((config.output_sigma, 0.01)),
        ),
    }

    # Choose best hypers
    if config.adaptive_gp_hypers is False:
        params_model, rmse = find_best_hypers(all_data, params_model)
        lengthscales_list: List[List[float]] = params_model["gp_params"]["ls"]
        alpha_list: List[List[float]] = params_model["gp_params"]["alpha"]
    sigma_list: List[List[float]] = [0.01, 0.01]

    ndpts = x_full.shape[0]

    for repeat in range(config.n_repeat):
        if config.run_distributed:
            assert config.n_repeat == 1
            seed = args.seed
        else:
            seed = repeat
        set_all_seeds(seed=seed)
        choices = np.random.choice(ndpts, config.n_initial)
        initial_noise = np.random.normal(0, config.output_sigma, (config.n_initial, dim_y))
        initial_noise_realspace = y_scaler.inverse_transform(initial_noise) - y_scaler.inverse_transform(
            np.zeros(initial_noise.shape)
        )

        x_start = x_full[choices]
        x_start_norm = x_full_norm[choices]

        y_start = y_full[choices] + initial_noise_realspace
        y_start_norm = y_full_norm[choices] + initial_noise

        algo_params = {
            "name": config.subregion_algo_name,
            "subregion_algo_params": config.subregion_algo_params,
            "x_normalizer": x_scaler,
            "y_normalizer": y_scaler,
        }

        subregion_algo = algorithm_type(name=config.subregion_algo_name, algo_params=algo_params)

        data = Namespace()
        data.x = [tuple(xi) for xi in x_start_norm]
        data.y = [tuple(yi) for yi in y_start_norm]
        data.x_unnorm = [tuple(xi) for xi in x_start]
        data.y_unnorm = [tuple(yi) for yi in y_start]

        # Define subregion algorithm

        metrics_object = EvaluationMetrics(
            subregion_algo,
            x_full=x_full,
            y_full=y_full,
            x_full_norm=x_full_norm,
            y_full_norm=y_full_norm,
            y_scaler=y_scaler,
            config=config,
        )

        collected_data_ids = list(choices)
        acqfn_all_iter = []

        X_tensor = tf.convert_to_tensor(x_full_norm, dtype=tf.float64)

        if config.mobo_maximize:
            y_tensor = -tf.convert_to_tensor(y_full_norm, dtype=tf.float64)
        else:
            y_tensor = tf.convert_to_tensor(y_full_norm, dtype=tf.float64)

        X_init = x_start_norm
        if config.mobo_maximize:
            Y_init = -y_start_norm
        else:
            Y_init = y_start_norm

        all_ids = set(np.arange(0, len(x_full_norm)))
        explored_ids = set(choices)
        unexplored_ids = all_ids - explored_ids

        for i_iter in tqdm(range(0, config.n_iter)):
            noise = np.random.normal(0, config.output_sigma, dim_y)
            noise_realspace = (
                y_scaler.inverse_transform(noise.reshape(1, -1))
                - y_scaler.inverse_transform(np.zeros(noise.shape).reshape(1, -1))
            )[0]

            if config.adaptive_gp_hypers and (i_iter % config.adaptive_gp_hypers_rate == 0):
                params_model, rmse = find_best_hypers(data, params_model, seed=repeat)
                lengthscales_list: List[List[float]] = params_model["gp_params"]["ls"]
                alpha_list: List[List[float]] = params_model["gp_params"]["alpha"]

            ExpData = Dataset(tf.constant(X_init), tf.constant(Y_init))

            model, gprs = build_trieste_model(
                data=ExpData,
                num_output=dim_y,
                lengthscales_list=lengthscales_list,
                alpha_list=alpha_list,
                sigma_list=sigma_list,
            )
            model: TrainableModelStack

            if config.prevent_requery:
                unexplored_ids = all_ids - explored_ids
                sliced_X_tensor = tf.gather(X_tensor, list(unexplored_ids))
            else:
                sliced_X_tensor = X_tensor

            parameter_space = DiscreteSearchSpace(sliced_X_tensor)
            result_ls = []
            for m in gprs:
                tmp = m.predict_f(x_full_norm)[0]
                result_ls.append(tmp.numpy().flatten())

            if config.mobo_maximize:
                posterior_mean = -np.array(result_ls).T
            else:
                posterior_mean = np.array(result_ls).T

            BO = EfficientGlobalOptimization(builder=ExpectedHypervolumeImprovement(), num_query_points=1)

            metrics_object.update_all_metrics(
                iteration_number=i_iter,
                collected_data_ids=collected_data_ids,
                posterior_mean=posterior_mean,
                acqfn_values=np.zeros(posterior_mean.shape),
                collected_data=data,
                gp_hypers=params_model,
            )

            x_next = BO.acquire_single(search_space=parameter_space, dataset=ExpData, model=model)

            distances = np.sqrt(np.sum((X_tensor - x_next) ** 2, axis=1))
            acquired_index_unexp = np.argmin(distances)

            y_next = y_tensor[acquired_index_unexp]
            y_next += noise

            explored_ids.add(acquired_index_unexp)
            unexplored_ids = all_ids - set(explored_ids)

            X_init = tf.concat([X_init, x_next], axis=0)
            Y_init = tf.concat([Y_init, tf.expand_dims(y_next, axis=0)], axis=0)

            collected_data_ids.append(acquired_index_unexp)
            data.x.append(tuple(x_full_norm[acquired_index_unexp]))
            data.x_unnorm.append(tuple(x_full[acquired_index_unexp]))

            data.y.append(tuple(y_full_norm[acquired_index_unexp]) + noise)
            data.y_unnorm.append(tuple(y_full[acquired_index_unexp]) + noise_realspace)

        # Check that all the points sampled are unique when we try to prevent requerying.
        if config.prevent_requery:
            assert len(set(collected_data_ids)) == (config.n_iter + len(choices))

        if config.run_distributed:
            repeat = args.seed

        output_name = "subregion_algo_{}_repeat_{}_strategy_EHVI".format(config.subregion_algo_name, repeat)

        # Create and dump data to pickle file
        if not os.path.exists(config.output_folder_path):
            os.makedirs(config.output_folder_path)
        with open(config.output_folder_path + output_name + ".pkl", "wb") as pickle_file:
            pickle.dump(metrics_object, pickle_file)
