import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge


class MedianModel:
    def __init__(self, median):
        self.median = median

    def predict(self, x):
        return np.full(x.size, self.median)


def get_task_model_map(resource_x, resource_y, training_data):
    task_model_map = {}
    for task in training_data["Task"].unique():
        task_training_data = training_data[training_data["Task"].apply(lambda s: s == task)][:6]
        x = task_training_data[resource_x].to_numpy()
        y = task_training_data[resource_y].to_numpy()
        if np.corrcoef(x, y)[0, 1] < 0.75:
            model = MedianModel(np.median(y))
        else:
            model = BayesianRidge(fit_intercept=True)
            model.fit(x.reshape(-1, 1), y)
        task_model_map[task] = model
    return task_model_map

