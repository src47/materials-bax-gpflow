import numpy as np
from scipy.stats import hypergeom

# Strategy class that can hold which strategy is happing as a function of iteration number

# Strategy Names
EXPLOITBAX = "MeanBAX"
REALEXPLOITBAX = 'RealExploitBAX'
INFOBAX = "InfoBAX"
US = "US"
RS = "RS"
HYBRIDBAX = 'SwitchBAX'

class StrategySelection:
    def __init__(
        self,
        current_strategy,
        future_strategy,
        number_of_iterations,
        percentage_strategy_allocation=1,
    ):
        self.current_strategy = current_strategy
        self.future_strategy = future_strategy
        self.number_of_iterations = number_of_iterations

        assert (percentage_strategy_allocation >= 0.0) and (
            percentage_strategy_allocation <= 1.0
        ), "Must be between 0 and 1"

        if self.future_strategy is None:
            self.percentage_strategy_allocation = 1.0
        else:
            self.percentage_strategy_allocation = percentage_strategy_allocation

    def switch_strategy(self, iteration_number):
        if self.future_strategy is not None:
            if iteration_number > int(
                (self.number_of_iterations) * self.percentage_strategy_allocation
            ):
                self.current_strategy = self.future_strategy

    def name_strategy(self):
        if self.future_strategy is not None:
            return (
                self.current_strategy
                + "_"
                + self.future_strategy
                + "_"
                + str(self.percentage_strategy_allocation)
            )
        else:
            return self.current_strategy


# switching condition for strategies
def switch_strategy(
    current_strategy, current_iteration, N_before_switch, new_strategy=["PM_US", 100]
):
    if current_iteration < N_before_switch:
        strategy = current_strategy
    else:
        strategy = new_strategy
    return strategy


# get acquisition functions as a function of x_domain for bax
def bax_acquisition_batched(x_domain, acqfn, batch_size=64):
    acqfn.initialize()
    x_batched = np.array_split(x_domain, batch_size)
    processed_batch = [acqfn.get_acq_list_batch(batch) for batch in x_batched]
    return np.concatenate(processed_batch)


# draw from the posterior of the gaussian process model
def get_posterior_mean_and_std(x, model):
    mu_pred, std_pred = model.get_post_mu_cov(x, full_cov=False)
    posterior_mean = np.array(mu_pred).T
    posterior_std = np.array(std_pred).T
    return posterior_mean, posterior_std


# Compute the acquisition function under various sampling strategies
def compute_acquisition_function(
    x_domain,
    acq_strategy,
    model=None,
    subregion_algo=None,
    bax_acqfn=None,
    y_scaler=None,
    collected_ids=None
):
    discrete_acquisition_function = np.zeros(x_domain.shape[0])
    if acq_strategy == RS:
        # one of these will be randomly maximal
        discrete_acquisition_function = np.random.uniform(0, 1, x_domain.shape[0])
    elif (acq_strategy == US) and (model is not None):
        posterior_mean, posterior_std = get_posterior_mean_and_std(x_domain, model)
        if y_scaler is not None:
            posterior_mean = y_scaler.inverse_transform(posterior_mean)
            posterior_std = y_scaler.inverse_transform(posterior_std)
        discrete_acquisition_function = np.mean(
            posterior_std, axis=1
        )  
    elif (
        (acq_strategy == EXPLOITBAX)
        and (model is not None)
        and (subregion_algo is not None)
    ):
        posterior_mean, posterior_std = get_posterior_mean_and_std(x_domain, model)
        if y_scaler is not None:
            posterior_mean = y_scaler.inverse_transform(posterior_mean)
            posterior_std = y_scaler.inverse_transform(posterior_std)            
        desired_x_idx_posterior_mean = subregion_algo.identify_subspace(
            x=x_domain, y=posterior_mean
        )
        if (set(desired_x_idx_posterior_mean).issubset(collected_ids)) or (len(desired_x_idx_posterior_mean) == 0):
            discrete_acquisition_function = np.mean(posterior_std, axis=1)
        else:
            discrete_acquisition_function[desired_x_idx_posterior_mean] = np.mean(posterior_std, axis=1)[desired_x_idx_posterior_mean]
    elif (
        (acq_strategy == HYBRIDBAX)
        and (model is not None)
        and (subregion_algo is not None)
    ):
        posterior_mean, posterior_std = get_posterior_mean_and_std(x_domain, model)
        if y_scaler is not None:
            posterior_mean = y_scaler.inverse_transform(posterior_mean)
            posterior_std = y_scaler.inverse_transform(posterior_std)            
        desired_x_idx_posterior_mean = subregion_algo.identify_subspace(
            x=x_domain, y=posterior_mean
        )
        if (set(desired_x_idx_posterior_mean).issubset(collected_ids)) or (len(desired_x_idx_posterior_mean) == 0):
            print('reached')
            discrete_acquisition_function = bax_acquisition_batched(x_domain, bax_acqfn)
        else:
            discrete_acquisition_function[desired_x_idx_posterior_mean] = np.mean(posterior_std, axis=1)[desired_x_idx_posterior_mean]
    # elif (
    #     (acq_strategy == REALEXPLOITBAX)
    #     and (model is not None)
    #     and (subregion_algo is not None)
    # ):
    #     posterior_mean, posterior_std = get_posterior_mean_and_std(x_domain, model)
    #     if y_scaler is not None:
    #         posterior_mean = y_scaler.inverse_transform(posterior_mean)
    #         posterior_std = y_scaler.inverse_transform(posterior_std)
    #     discrete_acquisition_function = np.mean(
    #         posterior_std, axis=1
    #     )  # predictors should be on similar scales
    #     desired_x_idx_posterior_mean = subregion_algo.identify_subspace(
    #         x=x_domain, y=posterior_mean
    #     )
    #     if desired_x_idx_posterior_mean != []:
    #         other_indices = list(
    #             set(np.arange(0, len(x_domain))) - set(desired_x_idx_posterior_mean)
    #         )
    #         discrete_acquisition_function[other_indices] = 1000000
        # discrete_acquisition_function = -discrete_acquisition_function
    elif (acq_strategy == INFOBAX) and (bax_acqfn is not None):
        discrete_acquisition_function = bax_acquisition_batched(x_domain, bax_acqfn)
    else:
        raise Exception("Invalid acquisition function")
    small_noise = 1e-8 * np.random.normal(
        loc=0, scale=1, size=discrete_acquisition_function.shape
    )
    discrete_acquisition_function += small_noise
    return discrete_acquisition_function


def optimize_discrete_acqusition_function(
    discrete_acquisition_function, zero_previous_queries=False, previous_indices=[]
):
    """Function to optimize acquisition function. Assumption is that the acquisition function has the same
    shape as the domain grid and the same indices. The function also allows to prevent requerying by zeroing
    the acquisition function at previous queries.

    Args:
        discrete_acquisition_function (_type_): _description_
        zero_previous_queries (bool, optional): _description_. Defaults to False.
        previous_indices (list, optional): _description_. Defaults to [].

    Returns:
        _type_: _description_
    """

    if zero_previous_queries and (previous_indices != []):
        discrete_acquisition_function[previous_indices] = np.min(
            discrete_acquisition_function
        )
        return np.argmax(discrete_acquisition_function)
    else:
        return np.argmax(discrete_acquisition_function)


def worst_case_performance(n_total, n_desired, n_experiments, n_specified=None):
    """calculate probability of getting n_desired points out of n_total points in n_experiments by sampling without replacement

    Args:
        N_total (_type_): number of total points in domain
        N_desired (_type_):number of points in domain that are desired
        N_experiments (_type_): number of experimental measurements

    Returns:
        prob: probability of getting all desired points
        mean: average number of desired points obtained
    """
    rv = hypergeom(M=n_total, n=n_desired, N=n_experiments)
    prob = rv.pmf(n_desired)
    x = np.arange(0, n_desired + 1)  # all possible number of desired points obtained
    pmf = rv.pmf(x)
    mean = np.sum(x * pmf)  # average number of desired points obtained
    return prob, mean


# Thompson sampling
