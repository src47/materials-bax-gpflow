import json
import numpy as np
import os
import pickle
from argparse import ArgumentParser, Namespace

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from subspace_algorithm.subspace_algorithm_utils.utils import (
    set_all_seeds,
    XY_from_csv,
    function_from_data,
)
from src.bax.models.gpfs_gp import BatchMultiGpfsGp, find_best_hypers
from src.bax.acq.acquisition import MultiBaxAcqFunction
from src.bax.util.misc_util import dict_to_namespace
from subspace_algorithm.subspace_algorithm_utils.sampling_strategies import (
    StrategySelection,
    compute_acquisition_function,
    optimize_discrete_acqusition_function,
)
from subspace_algorithm.subspace_algorithm_utils.metrics_evaluation import (
    EvaluationMetrics,
)
from subspace_algorithm.bax_subspace_algorithm_children import algorithm_type


model_class = BatchMultiGpfsGp

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

    acquisition_strategy = (
        config.strategy
    )  # acquisition strategy. E.g. Run RS then run US
    n_fithyper = config.n_iter  # fit hparams at every iteration until n_fithyper
    strategy_list = [
        strat[0] for strat in acquisition_strategy
    ]  # Sampling strategy names (BAX, US etc..)

    dim_x = len(config.columns_x)
    dim_y = len(config.columns_y)
    x_full, y_full = XY_from_csv(
        config.dataset_path, config.columns_x, config.columns_y
    )

    fn = function_from_data(
        x_full, y_full
    )  # convert X, y discrete file to a callable function

    # Construct Full Dataset Here
    all_data = Namespace()
    all_data.x_unnorm = [tuple(xi) for xi in x_full]
    all_data.y_unnorm = [tuple(yi) for yi in y_full]

    x_scaler = MinMaxScaler((0, 1))
    y_scaler = MinMaxScaler((-1, 1))

    x_scaler.fit(
        np.vstack((np.array(config.min_values_x), np.array(config.max_values_x)))
    )
    y_scaler.fit(
        np.vstack((np.array(config.min_values_y), np.array(config.max_values_y)))
    )

    x_full_norm = x_scaler.transform(x_full)
    y_full_norm = y_scaler.transform(y_full)

    all_data.x = [tuple(xi) for xi in x_full_norm]
    all_data.y = [tuple(yi) for yi in y_full_norm]

    ndpts = x_full.shape[0]

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

    for repeat in range(config.n_repeat):
        if config.run_distributed:
            assert config.n_repeat == 1
            seed = args.seed
        else:
            seed = repeat
        set_all_seeds(seed=seed)
        choices = np.random.choice(ndpts, config.n_initial)
        initial_noise = np.random.normal(
            0, config.output_sigma, (config.n_initial, dim_y)
        )
        initial_noise_realspace = y_scaler.inverse_transform(
            initial_noise
        ) - y_scaler.inverse_transform(np.zeros(initial_noise.shape))

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

        # Define subregion algorithm
        subregion_algo = algorithm_type(
            name=config.subregion_algo_name, algo_params=algo_params
        )

        acqfn_params = {"n_path": config.n_paths, "verbose": False}

        # Parallelizable
        for strategy in acquisition_strategy:
            data = Namespace()
            data.x = [tuple(xi) for xi in x_start_norm]
            data.y = [tuple(yi) for yi in y_start_norm]
            data.x_unnorm = [tuple(xi) for xi in x_start]
            data.y_unnorm = [tuple(yi) for yi in y_start]

            metrics_object = EvaluationMetrics(
                subregion_algo,
                x_full=x_full,
                y_full=y_full,
                x_full_norm=x_full_norm,
                y_full_norm=y_full_norm,
                y_scaler=y_scaler,
                config=config,
            )

            strategy_object = StrategySelection(
                current_strategy=strategy[0],
                future_strategy=strategy[1],
                number_of_iterations=config.n_iter,
                percentage_strategy_allocation=strategy[2],
            )
            collected_data_ids = list(choices)
            acqfn_all_iter = []

            for i_iter in tqdm(range(0, strategy_object.number_of_iterations)):
                noise = np.random.normal(0, config.output_sigma, dim_y)
                noise_realspace = (
                    y_scaler.inverse_transform(noise.reshape(1, -1))
                    - y_scaler.inverse_transform(np.zeros(noise.shape).reshape(1, -1))
                )[0]

                if config.adaptive_gp_hypers and (
                    i_iter % config.adaptive_gp_hypers_rate == 0
                ):
                    params_model, rmse = find_best_hypers(
                        data, params_model, seed=repeat
                    )

                model = BatchMultiGpfsGp(params_model, data, False)

                strategy_object.switch_strategy(i_iter)
                if (strategy_object.current_strategy == "InfoBAX") or (strategy_object.current_strategy == "SwitchBAX"):
                    subregion_algo.params.x_domain = x_full_norm
                    acqfn = MultiBaxAcqFunction(
                        params=acqfn_params, model=model, algorithm=subregion_algo
                    )
                else:
                    acqfn = None
                discrete_acquisition_function = compute_acquisition_function(
                    x_domain=x_full_norm,
                    acq_strategy=strategy_object.current_strategy,
                    model=model,
                    subregion_algo=subregion_algo,
                    bax_acqfn=acqfn,
                    y_scaler=y_scaler,
                    collected_ids=collected_data_ids
                )

                mu_pred, _ = model.get_post_mu_cov(all_data.x, full_cov=False)
                posterior_mean = np.array(mu_pred).T

                gprs = []

                result_ls = []
                for gpm in model.gpfsgp_list:
                    m = gpm.params.model
                    gprs.append(gpm.params.model)
                    tmp = m.predict_f(x_full_norm)[0]
                    result_ls.append(tmp.numpy().flatten())

                posterior_mean_gpflow = np.array(result_ls).T

                # run acquisition function
                id_next = optimize_discrete_acqusition_function(
                    discrete_acquisition_function,
                    zero_previous_queries=config.prevent_requery,
                    previous_indices=collected_data_ids,
                )

                metrics_object.update_all_metrics(
                    iteration_number=i_iter,
                    collected_data_ids=collected_data_ids,
                    posterior_mean=posterior_mean_gpflow,
                    acqfn_values=discrete_acquisition_function,
                    collected_data=data,
                    gp_hypers=params_model,
                )

                collected_data_ids.append(id_next)

                data.x.append(tuple(x_full_norm[id_next]))
                data.x_unnorm.append(tuple(x_full[id_next]))

                data.y.append(tuple(y_full_norm[id_next] + noise))
                data.y_unnorm.append(tuple(y_full[id_next] + noise_realspace))

                acqfn_all_iter.append(discrete_acquisition_function)

            # Check that all the points sampled are unique when we try to prevent requerying.
            if config.prevent_requery:
                assert len(set(collected_data_ids)) == (config.n_iter + len(choices))

            if config.run_distributed:
                repeat = args.seed

            output_name = "subregion_algo_{}_repeat_{}_strategy_{}".format(
                config.subregion_algo_name, repeat, strategy[0]
            )

            # metrics_object.dictionary_of_metrics["collected_data_ids"] = metrics_object.dictionary_of_metrics["collected_data_ids"][-1]
            # metrics_object.dictionary_of_metrics["collected_data"] = metrics_object.dictionary_of_metrics["collected_data"][-1]

            # Create and dump data to pickle file
            if not os.path.exists(config.output_folder_path):
                os.makedirs(config.output_folder_path)
            with open(
                config.output_folder_path + output_name + ".pkl", "wb"
            ) as pickle_file:
                pickle.dump(metrics_object, pickle_file)
