import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from TraceReader import TraceReader
from NodeFactor import get_node_factor_map
from Lotaru import get_task_model_map

if __name__ == '__main__':
    local_dir = os.getcwd()
    nodes = ["asok01", "asok02", "c2", "local", "n1", "n2", "wally"]

    parser = argparse.ArgumentParser(prog='lotaru2')
    parser.add_argument('-b', '--benchmark_dir', default=os.path.join("data", "benchmarks"))
    parser.add_argument('-e', '--experiment_number', default="1")
    parser.add_argument('-s', '--scale', action='store_true', default=False)
    parser.add_argument('-t', '--trace_dir', default=os.path.join("data", "traces"))
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('-w', '--workflow', default="eager")
    parser.add_argument('-x', '--resource_x', default="TaskInputSizeUncompressed")
    parser.add_argument('-y', '--resource_y', default="%cpu")
    args = parser.parse_args()
    for arg in vars(args).items():
        print(arg)

    trace_reader = TraceReader(args.trace_dir)
    training_data = trace_reader.read_training_data(args.workflow, args.experiment_number)
    tasks = training_data["Task"].unique()

    task_model_map = get_task_model_map(args.resource_x, args.resource_y, training_data)
    if args.scale:
        node_factor_map = get_node_factor_map(args.benchmark_dir)

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
