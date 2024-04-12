import os
import sys
import argparse
from functools import wraps

import numpy as np
import matplotlib.pyplot as plt

from TraceReader import TraceReader
from LotaruInstance import MedianModel
from RunExperiment import run_experiment

class AnalysisScript:
    def __init__(self, name, description, func):
        self.name = name
        self.description = description
        self.func = func

analysis_scripts = []

def toBool(s):
    if not (s == "True" or s == "False"):
        raise RuntimeError
    return s == "True"

# decorators break my head
# so what is supposed to happen?
#
# @option(...)
# def my_func():
#   ...
#
# translates to the following:
#
# def my_func():
#   ...
# my_func = option(...)(my_func)
#
# so option(...) has to return a function that takes my_func as an argument
# and then returns another function that we will call when we call my_func
#
# my current idea is as follows:
# I want to be able to call the resulting my_func with an arg_parser and
# argument_string as arguments like so: my_func(arg_parser, argument_string)
# each option decorator shall add one option to the arg_parser
# and a final @run decorator shall parse the argument_string with the
# arg_parser we just constructed and then call the original my_func with
# the parsed arguments
#
# does that work?
"""
@option(...)
@analysis
def func:
    ...

is the same as

def func:
    ...
setup = option(...)
f1 = analysis(func)
func = setup(f1)

func(parser, args) becomes the func_to_return from  setup, it adds an option to the parser
it then call the func_to_return from analysis which parses the arguments and call the
original function

"""

def register(func):
    analysis_scripts.append(AnalysisScript(func.__name__, func.__doc__, func))
    return func


def option(*args, **kwargs):
    def setup(func):
        @wraps(func)
        def func_to_return(arg_parser, arg_string):
            arg_parser.add_argument(*args, **kwargs)
            func(arg_parser, arg_string)
        return func_to_return
    return setup


def analysis(func):
    @wraps(func)
    def func_to_return(arg_parser, arg_string):
        args = arg_parser.parse_args(arg_string)
        func(args)
    return func_to_return


@register
@option("-e", "--experiment_number", default="1")
@analysis
def node_error(args):
    """
    returns the median relative prediction error
    for each node over all workflows and tasks
    """
    print("node_error was called with: ", args)
    results = run_experiment(experiment_number=args.experiment_number)
    # TODO row is misleading, this is a dataframe?
    def median_error(row):
        return np.median(np.abs(row["y"] - row["yhat"]) / row["yhat"])
    median_errors = results.groupby("node").apply(median_error)
    print(median_errors)


@register
@analysis
def results_csv(args):
    """
    writes the predictions for all workflows, tasks, nodes, and experiment_numbers
    to stdout. Output format is as follows:

    workflow;task;node;x;y
    """
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



@register
@analysis
def workflow_node_error(args):
    """
    creates one figure for each workflow
    each figure shows boxplots for each node
    each boxplot shows the distribution of the relative absolute error
    over all the traces of the given workflow that ran on the given node
    """
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

@register
@option("-w", "--workflow", default="eager")
@option("--scale-bayesian-model",  type=toBool, default=True)
@option("--scale-median-model", type=toBool, default=False)
@option('-x', '--resource-x', default="TaskInputSizeUncompressed")
@option('-y', '--resource-y', default="Realtime")
@analysis
def node_task_error(args):
    """
    creates a plot showing the average relative error
    for each task on different nodes for the given workflow
    """
    results = run_experiment(
            resource_x=args.resource_x,
            resource_y=args.resource_y,
            scale_bayesian_model=args.scale_bayesian_model,
            scale_median_model=args.scale_median_model,
            workflows=[args.workflow])
    nodes = results["node"].unique()
    def average_relative_error(x):
        return np.mean(np.abs(x["y"] - x["yhat"]) / x["yhat"])
    grouped = results.groupby(["node", "task"]).apply(average_relative_error)
    for node in nodes:
        task_err_map = grouped[node].to_dict()
        plt.scatter(task_err_map.keys(), task_err_map.values())
    plt.xticks(rotation=-45, ha='left')
    plt.legend(nodes)
    plt.show()


@register
@option("--scale", choices=["log", "linear"], default="log")
@analysis
def scale_median_model(args):
    results_scaled = run_experiment(scale_median_model=True)
    results_unscaled = run_experiment(scale_median_model=False)
    def get_errors(df):
        return df[df["model"] == MedianModel].apply(lambda row: np.abs(row["y"] - row["yhat"]) / row["yhat"], axis=1)
    errors_scaled = get_errors(results_scaled)
    errors_unscaled = get_errors(results_unscaled)
    plt.yscale(args.scale)
    plt.boxplot([list(errors_scaled), list(errors_unscaled)], labels=["scaled", "unscaled"])
    plt.show()


@register
@option("-w", "--workflow", default="eager")
@option("-e", "--experiment-number", default="1")
@analysis
def training_traces(args):
    trace_reader = TraceReader(os.path.join("data", "traces"))
    all_data = trace_reader.get_trace(args.workflow.lower(), "local")
    if args.experiment_number == "0":
        all_training_data = all_data[all_data["Label"].apply(lambda s: s[:6] == "train-")]
    else:
        all_training_data = all_data[all_data["Label"] == ("train-" + args.experiment_number)]
    results = all_training_data.groupby(["Workflow", "WorkflowInputSize", "Task"]).count()
    results.drop(labels=results.columns[1:], axis=1, inplace=True)
    results.columns = ["count"]
    print("unique number of traces per workflow workflow input size and task: ", results["count"].unique())

