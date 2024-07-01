class FactorModel:
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return (x * self.value).flatten()


class NaiveInstance:
    def __init__(self, training_data):
        self.training_data = training_data
        self.tasks = self.training_data.keys()
        self.task_model_map = {}

    def train_models(self):
        for task in self.tasks:
            x = self.training_data[task]["x"].to_numpy()
            y = self.training_data[task]["y"].to_numpy()
            self.task_model_map[task] = FactorModel((y / x).mean())

    def get_prediction(self, task, node, x):
        model = self.task_model_map[task]
        return model.predict(x.reshape(-1, 1))

    def get_model_for_task(self, task):
        return self.task_model_map[task]
