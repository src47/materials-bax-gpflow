"""

Contains functions and class to get metrics of sampling. 

"""


class EvaluationMetrics:

    """
    Class stores the performance / metrics of a sampling method as a function of iteration number.
    """

    def __init__(
        self, subregion_algo, x_full, y_full, x_full_norm, y_full_norm, y_scaler, config
    ):
        self.subregion_algo = subregion_algo
        self.x_full = x_full
        self.y_full = y_full
        self.x_full_norm = x_full_norm
        self.y_full_norm = y_full_norm
        self.y_scaler = y_scaler

        self.desired_x_idx_true = self.subregion_algo.identify_subspace(
            x=x_full, y=y_full
        )
        self.desired_x_true = x_full[self.desired_x_idx_true]
        self.desired_y_true = y_full[self.desired_x_idx_true]

        self.config = config
        self.dictionary_of_metrics = {
            "n_obtained": [],
            "n_obtained_posterior": [],
            "n_obtained_rs_wo_replacement": [],
            "precision_real": [],
            "recall_real": [],
            "precision_posterior": [],
            "jaccard_real": [],
            "jaccard_posterior": [],
            "recall_posterior": [],
            "collected_data_ids": [],
            "x_full": list(self.x_full),
            "y_full": list(self.y_full),
            "desired_x_idx_posterior_mean": [],
            "desired_x_true": list(self.desired_x_true),
            "desired_y_true": list(self.desired_y_true),
            "desired_x_idx_true": list(self.desired_x_idx_true),
            "posterior_mean": [],
            "acqfn_values": [],
            "collected_data": None,
            "gp_hypers": [],
        }

    @staticmethod
    def get_intersection(s_true, s_measured):
        return list(set(s_true).intersection(set(s_measured)))

    @staticmethod
    def get_precision(s_true, s_measured):
        if len(s_measured) == 0:
            return 0
        else:
            return len(EvaluationMetrics.get_intersection(s_true, s_measured)) / len(
                s_measured
            )

    @staticmethod
    def get_recall(s_true, s_measured):
        if len(s_true) == 0:
            return 0
        else:
            return len(EvaluationMetrics.get_intersection(s_true, s_measured)) / len(
                s_true
            )

    @staticmethod
    def get_n_target(s_true, s_measured):
        return len(EvaluationMetrics.get_intersection(s_true, s_measured))

    @staticmethod
    def get_jaccard(s_true, s_measured):
        if len(list(set(s_true).union(set(s_measured)))) == 0:
            return 0
        else:
            return len(EvaluationMetrics.get_intersection(s_true, s_measured)) / (
                len(list(set(s_true).union(set(s_measured))))
            )

    def get_n_obtained_expected_rs_wo_replacement(self, iteration_number):
        return iteration_number * len(self.desired_x_idx_true) / len(self.x_full)

    def get_fraction_repeated_queries(self):
        current_list_of_collected_data_ids = self.dictionary_of_metrics[
            "collected_data_ids"
        ][-1]
        number_of_collected_data_ids = len(current_list_of_collected_data_ids)
        number_of_collected_data_ids_not_repeated = len(
            set(current_list_of_collected_data_ids)
        )
        number_repeated = (
            number_of_collected_data_ids - number_of_collected_data_ids_not_repeated
        )
        return number_repeated / number_of_collected_data_ids

    def update_all_metrics(
        self,
        iteration_number,
        collected_data_ids,
        posterior_mean,
        acqfn_values,
        collected_data,
        gp_hypers,
    ):
        if self.y_scaler is not None:
            posterior_mean = self.y_scaler.inverse_transform(posterior_mean)

        desired_x_idx_posterior_mean = self.subregion_algo.identify_subspace(
            x=self.x_full, y=posterior_mean
        )

        n_obtained = EvaluationMetrics.get_n_target(
            self.desired_x_idx_true, collected_data_ids
        )
        n_obtained_posterior = EvaluationMetrics.get_n_target(
            self.desired_x_idx_true, desired_x_idx_posterior_mean
        )

        n_obtained_rs_wo_replacement = self.get_n_obtained_expected_rs_wo_replacement(
            iteration_number
        )

        precision_real = EvaluationMetrics.get_precision(
            self.desired_x_idx_true, collected_data_ids
        )
        precision_posterior = EvaluationMetrics.get_precision(
            self.desired_x_idx_true, desired_x_idx_posterior_mean
        )

        recall_real = EvaluationMetrics.get_recall(
            self.desired_x_idx_true, collected_data_ids
        )
        recall_posterior = EvaluationMetrics.get_recall(
            self.desired_x_idx_true, desired_x_idx_posterior_mean
        )

        jaccard_real = EvaluationMetrics.get_jaccard(
            self.desired_x_idx_true, collected_data_ids
        )
        jaccard_posterior = EvaluationMetrics.get_jaccard(
            self.desired_x_idx_true, desired_x_idx_posterior_mean
        )

        self.dictionary_of_metrics["n_obtained"].append(n_obtained)
        self.dictionary_of_metrics["n_obtained_rs_wo_replacement"].append(
            n_obtained_rs_wo_replacement
        )
        self.dictionary_of_metrics["n_obtained_posterior"].append(n_obtained_posterior)

        self.dictionary_of_metrics["precision_real"].append(precision_real)
        self.dictionary_of_metrics["precision_posterior"].append(precision_posterior)
        self.dictionary_of_metrics["recall_real"].append(recall_real)
        self.dictionary_of_metrics["recall_posterior"].append(recall_posterior)
        self.dictionary_of_metrics["jaccard_real"].append(jaccard_real)
        self.dictionary_of_metrics["jaccard_posterior"].append(jaccard_posterior)

        self.dictionary_of_metrics["collected_data_ids"].append(
            list(collected_data_ids)
        )
        self.dictionary_of_metrics["posterior_mean"].append(list(posterior_mean))
        self.dictionary_of_metrics["acqfn_values"].append(list(acqfn_values))
        self.dictionary_of_metrics["collected_data"] = collected_data
        self.dictionary_of_metrics["desired_x_idx_posterior_mean"].append(
            list(desired_x_idx_posterior_mean)
        )
        self.dictionary_of_metrics["gp_hypers"].append(gp_hypers)
