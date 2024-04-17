import os

import numpy as np
from sklearn.linear_model import BayesianRidge

from lotaru.NodeFactor import get_node_factor_map


class MedianModel:
    def __init__(self, median):
        self.median = median

    def predict(self, x):
        return np.full(x.size, self.median)


class LotaruInstance:
    def __init__(self, training_data, scale_bayesian_model, scale_median_model):
        self.training_data = training_data
        self.scale_bayesian_model = scale_bayesian_model
        self.scale_median_model = scale_median_model
        self.tasks = self.training_data.keys()
        if scale_bayesian_model or scale_median_model:
            self.node_factor_map = get_node_factor_map(os.path.join("data", "benchmarks"))
        self.task_model_map = {}

    def train_models(self):
        for task in self.tasks:
            x = self.training_data[task]["x"].to_numpy()
            y = self.training_data[task]["y"].to_numpy()
            pearson = np.corrcoef(x, y)[0, 1]
            if np.isnan(pearson) or pearson < 0.75:
                model = MedianModel(np.median(y))
            else:
                model = BayesianRidge(fit_intercept=True)
                model.fit(x.reshape(-1, 1), y)
            self.task_model_map[task] = model

    def get_prediction(self, task, node, x):
        model = self.task_model_map[task]
        if (self.scale_bayesian_model and type(model) is BayesianRidge) or (
                self.scale_median_model and type(model) is MedianModel):
            factor = self.node_factor_map[node]
        else:
            factor = 1
        return model.predict(x.reshape(-1, 1)) * factor

    def get_tasks(self):
        return self.tasks

    def get_model_for_task(self, task):
        return self.task_model_map[task]
