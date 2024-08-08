import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from lotaru.analysis.analysis_script import (
    register, option, defaultanalysis, analysis)
from lotaru.TraceReader import TraceReader
from lotaru.LotaruInstance import MedianModel
from lotaru.RunExperiment import run_experiment
from lotaru.Constants import WORKFLOWS, NODES

registered_scripts = []


@register(registered_scripts)
@defaultanalysis
def node_error(args, results):
    """
    Returns the median relative prediction error for each node, over all
    workflows and tasks.
    """
    median_errors = results.groupby("node")["rae"].median()
    print(median_errors.sort_index())


@register(registered_scripts)
@option('--output-file', default='-')
@defaultanalysis
def results_csv(args, results):
    """
    Writes the predictions for all workflows, tasks, nodes, and given
    experiment numbers to a given file or stdout. If '-o' is not specified or
    '-o -' is given the predictions are written to stdout. In any other case
    the argument to '-o' is treated as the path of the file to write to. This
    script refuses to overwrite existing files.

    Output format is as follows:

    workflow;task;node;x;y;yhat;rae

    """
    out = results.to_csv(sep=";",
                         columns=["workflow", "task", "node",
                                  "x", "y", "yhat", "rae"],
                         header=True, index=False)
    if args.output_file == "-":
        print(out)
        return
    if os.path.isfile(args.output_file):
        print("refusing to overwrite existing file", file=sys.stderr)
        exit(-1)
    with open(args.output_file, "w") as file:
        file.write(out)


@register(registered_scripts)
@option('-s', '--save', default="")
@defaultanalysis
def workflow_node_error(args, results):
    """
    Visualizes the distribution of relative absolute error over all traces
    nodes and workflows.

    Creates one figure for each workflow. Each figure shows boxplots for each
    node and each boxplot shows the distribution of the relative absolute error
    over all the traces of the given workflow that ran on the given node.
    """
    # transform into data structure that can be plotted
    data = {}
    for workflow in WORKFLOWS:
        data[workflow] = []
        for node in NODES:
            data[workflow].append(
                results.loc[(results["workflow"] == workflow) &
                            (results["node"] == node), "rae"])

    fig, axs = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
    axs = axs.flatten()
    for i in range(len(WORKFLOWS)):
        axs[i].set_title(WORKFLOWS.keys[i])
        axs[i].boxplot(data[WORKFLOWS.keys[i]])
        axs[i].set_xticklabels(NODES, rotation=-45, ha='left')
    axs[0].set_yscale("log")
    axs[0].set_ylim(bottom=results.loc[results["rae"] > 0, "rae"].min())
    axs[0].set_ylabel("relative absolute prediction error")
    axs[2].set_xlabel("nodes")
    fig.tight_layout()
    if args.save != "":
        plt.savefig(args.save)
    else:
        plt.show()


@register(registered_scripts)
@option('--save', default="")
@option("-w", "--workflow", default="eager")
@defaultanalysis
def node_task_error(args, results):
    """
    Creates a plot showing the average relative error for each task and node
    for the given workflow.
    """
    results = results[results["workflow"] == args.workflow]

    grouped = results.groupby(["node", "task"])["rae"].mean()
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    for node in NODES:
        task_err_map = grouped[node].to_dict()
        axs.scatter(task_err_map.keys(), task_err_map.values())
    axs.set_xticklabels(list(task_err_map.keys()), rotation=-90, ha='left')
    axs.set_ylabel("average relative error")
    axs.set_xlabel("task")
    axs.set_title(
        f"average relative error per task and node for {args.workflow}")
    axs.legend(NODES)
    fig.tight_layout()
    if args.save != "":
        plt.savefig(args.save)
    else:
        plt.show()


@register(registered_scripts)
@option("--scale", choices=["log", "linear"], default="log")
@option('-s', '--save', default="")
@analysis
def scale_median_model(args):
    '''
    Answers the question if lotaru should scale the outputs of its median
    models. Shows the distribution of errors of predictions made with scaled
    median models and unscaled median models.
    '''
    results_scaled = run_experiment(scale_median_model=True)
    results_unscaled = run_experiment(scale_median_model=False)

    def get_errors(df):
        return df[df["model"] == MedianModel].apply(
            lambda row: np.abs(row["y"] - row["yhat"]) / row["yhat"],
            axis=1)
    errors_scaled = get_errors(results_scaled)
    errors_unscaled = get_errors(results_unscaled)
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.boxplot([list(errors_scaled), list(errors_unscaled)],
                labels=["scaled", "unscaled"])
    axs.set_ylabel("relative absolute error")
    axs.set_yscale(args.scale)
    axs.set_xlabel("type of median model")
    axs.set_title("performace of scaled and unscaled median models")
    fig.tight_layout()
    if args.save != "":
        plt.savefig(args.save)
    else:
        plt.show()


@register(registered_scripts)
@option("-w", "--workflow", default="eager")
@option("-e", "--experiment-number", default="1")
@analysis
def training_traces(args):
    '''
    Let x be the number of instances of a given task during one workflow
    execution. This script prints the unique values of x for all tasks and
    workflow executions in our traces.

    For example if x is [1] that means that during all workflow exections each
    task had only one instance. If x is [1, 2] that means some tasks during
    some workflow executions had two instances while the others only had one.
    '''
    trace_reader = TraceReader(os.path.join("data", "traces"))
    all_data = trace_reader.get_trace(args.workflow.lower(), "local")
    if args.experiment_number == "0":
        all_training_data = all_data[all_data["label"].apply(
            lambda s: s[:6] == "train-")]
    else:
        all_training_data = all_data[all_data["label"] == (
            "train-" + args.experiment_number)]
    results = all_training_data.groupby(
        ["workflow", "workflowinputsize", "task"]).count()
    results.drop(labels=results.columns[1:], axis=1, inplace=True)
    results.columns = ["count"]
    print("unique number of traces per workflow workflow input size and task: ",
          results["count"].unique())
