"""
Predicting RUntimes from Cpu and IO
more precisely: locally predicting workflow task runtimes
by using the uncompressed task input size to predict task
cpu and io and using cpu and io to predict task runtimes
"""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge

from LotaruInstance import LotaruInstance
from TraceReader import TraceReader

nodes = ["asok01", "asok02", "n1", "n2", "c2", "local"]
workflow = "eager"
experiment_number = "1"
resource_x = "TaskInputSizeUncompressed"
trace_reader = TraceReader(os.path.join("data", "traces"))
training_data = trace_reader.get_training_data(workflow, experiment_number)
tasks = training_data["Task"].unique()


def get_training_data(task, cpuLotaru, ioLotaru):
    """
    returns the training data as two numpy ndarrays x and y
    the x values are either taken directly from the training set
    or predicted using the given models
    both give similar results
    """
    assert bool(cpuLotaru) == bool(ioLotaru)
    task_training_data = training_data[training_data["Task"] == task][:6]
    if bool(cpuLotaru):
        xin = task_training_data[resource_x].to_numpy()
        x1 = cpuLotaru.get_prediction(task, "local", xin.reshape(-1, 1))
        x2 = ioLotaru.get_prediction(task, "local", xin.reshape(-1, 1))
        x = np.array(list(zip(x1, x2)))
    else:
        x = task_training_data[["%cpu", "rchar"]].to_numpy()
    y = task_training_data["Realtime"].to_numpy()
    return (x.reshape(-1, 2), y)


def get_prucio_predictions():

    # train models
    cpuLotaru = LotaruInstance(workflow, experiment_number, resource_x,
                               "%cpu", trace_reader, False, False)
    cpuLotaru.train_models()

    ioLotaru = LotaruInstance(workflow, experiment_number, resource_x,
                              "rchar", trace_reader, False, False)
    ioLotaru.train_models()

    task_model_map = {}
    for task in tasks:
        x, y = get_training_data(task, None, None)
        # pearson = np.corrcoef(x.transpose(), y)
        # p1 = pearson[-1, 0]
        # p2 = pearson[-1, 1]
        model = BayesianRidge()
        model.fit(x, y)
        task_model_map[task] = model

    # predict
    results = pd.DataFrame(
        columns=["workflow", "task", "node", "x", "yhat", "y"])
    for task in tasks:
        for node in nodes:
            test_data = trace_reader.get_test_data(workflow, task, node)
            x = test_data[resource_x].to_numpy()
            yhat = test_data["Realtime"].to_numpy()
            x1 = cpuLotaru.get_prediction(task, node, x.reshape(-1, 1))
            x2 = ioLotaru.get_prediction(task, node, x.reshape(-1, 1))
            y = task_model_map[task].predict(
                np.array(list(zip(x1, x2))).reshape(-1, 2))
            for i in range(x.size):
                results.loc[results.index.size] = [
                    workflow, task, node, x[i], yhat[i], y[i]]

    return results


results = get_prucio_predictions()


def median_error(row):
    return np.median(np.abs(row["y"] - row["yhat"]) / row["yhat"])


median_errors = results.groupby("node").apply(median_error)
print(median_errors)
