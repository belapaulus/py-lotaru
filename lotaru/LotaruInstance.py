import numpy as np
from sklearn.linear_model import BayesianRidge


class MedianModel:
    def __init__(self, median):
        self.median = median

    def predict(self, x):
        return np.full(x.size, self.median)


class LotaruInstance:
    def __init__(self, training_data, scaler, scale_bayesian_model=True,
                 scale_median_model=False):
        self.training_data = training_data
        self.tasks = self.training_data.keys()
        self.scale_model = {
            BayesianRidge: scale_bayesian_model,
            MedianModel: scale_median_model,
        }
        self.scaler = scaler
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
        if self.scale_model[type(model)]:
            factor = self.scaler.get_factor(node, task)
        else:
            factor = 1
        return model.predict(x.reshape(-1, 1)) * factor

    def get_model_for_task(self, task):
        return self.task_model_map[task]
