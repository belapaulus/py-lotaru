import numpy as np
import pandas as pd
from scipy import stats


class OnlineModel:
    def __init__(self, x, y):
        self.training_data = pd.DataFrame({"x": x, "y": y})

    def predict(self, x):
        ratios = np.apply_along_axis(
            lambda a: self.get_ratio(a), 1, x).reshape(-1, 1)
        return (x * ratios).flatten()

    def get_ratio(self, x):
        x = x[0]
        distance = np.abs(self.training_data["x"] - x)
        closest = self.training_data.iloc[distance.idxmin()]
        return closest["y"] / closest["x"]


class MedianModel:
    def __init__(self, x, y):
        self.median = np.median(y)

    def predict(self, x):
        return np.full(x.size, self.median)


class KSModel:
    def __init__(self, x, y):
        self.value = self._get_value(y)

    def _get_value(self, y):
        std = y.std(ddof=1)
        if std == 0:
            return np.median(y)
        norm = stats.norm(loc=y.mean(), scale=std)
        if stats.kstest(y, norm.cdf).pvalue > self._critical_value(len(y)):
            return norm.rvs()
        gamma = stats.gamma(1, loc=y.mean())
        if stats.kstest(y, gamma.cdf).pvalue > self._critical_value(len(y)):
            return gamma.rvs()
        return np.median(y)

    def _critical_value(self, length):
        default = 0.238
        lookup = {
            1: 0.950,
            2: 0.776,
            3: 0.636,
            4: 0.565,
            5: 0.510,
            6: 0.468,
            7: 0.436,
            8: 0.410,
            9: 0.387,
            10: 0.369,
            11: 0.352,
            12: 0.338,
            13: 0.325,
            14: 0.314,
            15: 0.304,
            16: 0.295,
            17: 0.286,
            18: 0.279,
            19: 0.271,
            20: 0.265,
            21: 0.259,
            22: 0.253,
            23: 0.247,
            24: 0.242,
        }
        if length in lookup:
            return lookup[length]
        return default

    def predict(self, x):
        return np.full(x.size, self.value)


class OnlineInstance:
    def __init__(self, training_data, alt_model_type):
        alt_models = {
            "p": KSModel,
            "m": MedianModel,
        }
        self.alt_model = alt_models[alt_model_type]
        self.training_data = training_data
        self.tasks = self.training_data.keys()
        self.task_model_map = {}

    def train_models(self):
        for task in self.tasks:
            x = self.training_data[task]["x"].to_numpy()
            y = self.training_data[task]["y"].to_numpy()
            pearson = np.corrcoef(x, y)[0, 1]
            if np.isnan(pearson) or pearson < 0.8:
                model = self.alt_model(x, y)
            else:
                model = OnlineModel(x, y)
            self.task_model_map[task] = model

    def get_prediction(self, task, node, x):
        return self.task_model_map[task].predict(x.reshape(-1, 1))

    def get_model_for_task(self, task):
        return self.task_model_map[task]
