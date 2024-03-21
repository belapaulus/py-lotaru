import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from RunExperiment import run_experiment

# TODO row is misleading, this is a dataframe?
def median_error(row):
    return np.median(np.abs(row["y"] - row["yhat"]) / row["yhat"])

# returns the median relative prediction error
# for each node over all workflows and tasks
def node_error():
    results = run_experiment(experiment_number="2")
    median_errors = results.groupby("node").apply(median_error)
    print(median_errors)


# writes predictions for all workflows, tasks and inputs to csv files
def results_csv():
    for i in [1, 2]:
        results = run_experiment(experiment_number=str(i))
        def print_row(row):
            print(";".join([
                row["workflow"],
                row["task"].lower(),
                row["node"],
                str(int(row["x"])),
                str(int(row["y"]))]))
        results.apply(print_row, axis=1)



# creates one figure for each workflow
# each figure shows boxplots for each node
# each boxplot shows the distribution of the relative absolute error
# over all the traces of the given workflow that ran on the given node
def workflow_node_error():
    results = run_experiment()
    workflows = results["workflow"].unique()
    nodes = results["node"].unique()
    def relative_absolute_error(x):
        return np.abs(x["y"] - x["yhat"]) / x["yhat"]
    grouped = results.groupby(["workflow", "node"]).apply(relative_absolute_error) 
    plt.figure()
    num_rows = 2
    num_cols = 3
    plt.subplot(num_rows, num_cols, 1)
    for i in range(len(workflows)):
        workflow = workflows[i]
        plt.subplot(num_rows, num_cols, i+1)
        plt.yscale("log")
        plt.title(workflow)
        data = []
        for node in nodes:
            data.append(grouped[(workflow, node)].to_numpy())
        plt.boxplot(data)

    plt.show()

