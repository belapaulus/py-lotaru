import os

import numpy as np
from sklearn.linear_model import BayesianRidge

from NodeFactor import get_node_factor_map


class MedianModel:
    def __init__(self, median):
        self.median = median

    def predict(self, x):
        return np.full(x.size, self.median)


class LotaruInstance:
    def __init__(self, workflow, experiment_number, resource_x, resource_y, trace_reader,
                 scale_bayesian_model, scale_median_model):
        self.workflow = workflow
        self.experiment_number = experiment_number
        self.resource_x = resource_x
        self.resource_y = resource_y
        self.trace_reader = trace_reader
        self.scale_bayesian_model = scale_bayesian_model
        self.scale_median_model = scale_median_model

        self.training_data = trace_reader.get_training_data(workflow, experiment_number)
        self.tasks = self.training_data["Task"].unique()
        self.task_training_data_map = {}
        for task in self.tasks:
            self.task_training_data_map[task] = self.training_data[self.training_data["Task"].apply(
                lambda s: s == task)][:6]

        if scale_bayesian_model or scale_median_model:
            self.node_factor_map = get_node_factor_map(os.path.join("data", "benchmarks"))

        self.task_model_map = {}

    def train_models(self):
        self.task_model_map = {}
        for task in self.tasks:
            x = self.task_training_data_map[task][self.resource_x].to_numpy()
            y = self.task_training_data_map[task][self.resource_y].to_numpy()
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
