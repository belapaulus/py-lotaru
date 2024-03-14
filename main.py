import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from TraceReader import TraceReader


class MedianModel:
    def __init__(self, median):
        self.median = median

    def predict(self, x):
        return x * 0 + self.median


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


def get_node_factor_map(nodes):
    scores = pd.read_csv(local_dir + "/benchmarkScores.csv", index_col=0, dtype={"node": "string"})
    local_cpu = scores.loc["wally", "cpu_score"]
    local_io = scores.loc["wally", "io_score"]
    node_factor_map = {}
    for node in nodes:
        node_factor_map[node] = ((local_cpu / scores.loc[node, "cpu_score"]) + (
                local_io / scores.loc[node, "io_score"])) / 2
    return node_factor_map


if __name__ == '__main__':
    local_dir = os.getcwd()
    nodes = ["asok01", "asok02", "c2", "local", "n1", "n2", "wally"]

    parser = argparse.ArgumentParser(prog='lotaru2')
    parser.add_argument('-t', '--trace_dir', default=os.path.join(local_dir, "traces"))
    parser.add_argument('-w', '--workflow', default="eager")
    parser.add_argument('-x', '--resource_x', default="TaskInputSizeUncompressed")
    parser.add_argument('-y', '--resource_y', default="%cpu")
    parser.add_argument('-s', '--scale', action='store_true', default=False)
    parser.add_argument('-e', '--experiment_number', default="1")
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    args = parser.parse_args()
    for arg in vars(args).items():
        print(arg)

    trace_reader = TraceReader(args.trace_dir)
    training_data = trace_reader.read_training_data(args.workflow, args.experiment_number)
    tasks = training_data["Task"].unique()

    task_model_map = get_task_model_map(args.resource_x, args.resource_y, training_data)
    if args.scale:
        node_factor_map = get_node_factor_map(nodes)

    # predict and evaluate
    for node in nodes:
        task_err_map = {}
        for task in tasks:
            test_data = trace_reader.read_test_data(args.workflow, node)
            task_test_data = test_data[test_data["Task"].apply(lambda s: s == task)]
            x = task_test_data[args.resource_x].to_numpy()
            yhat = task_test_data[args.resource_y].to_numpy()
            y = task_model_map[task].predict(x.reshape(-1, 1))
            if args.scale:
                y = y * node_factor_map[node]
            # print(y, yhat, np.abs(y-yhat), np.abs(y-yhat) / yhat, np.mean(np.abs(y-yhat) / yhat))
            average_relative_error = np.mean(np.abs(y - yhat) / yhat)
            task_err_map[task] = average_relative_error
        plt.scatter(task_err_map.keys(), task_err_map.values())

    plt.xticks(rotation=-45, ha='left')
    plt.legend(nodes)
    plt.title(" ".join([args.workflow, args.resource_x, args.resource_y, str(args.scale), args.experiment_number]))
    plt.show()
