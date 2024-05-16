import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from lotaru.analysis.analysis_script import AnalysisScript, register, option, analysis, toBool
from lotaru.TraceReader import TraceReader
from lotaru.LotaruInstance import MedianModel
from lotaru.RunExperiment import run_experiment

registered_scripts = []


@register(registered_scripts)
@option("-e", "--experiment_number", default="1")
@option("--scale-bayesian-model",  type=toBool, default=True)
@option("--scale-median-model", type=toBool, default=False)
@option('-x', '--resource-x', default="TaskInputSizeUncompressed")
@option('-y', '--resource-y', default="Realtime")
@analysis
def node_error(args):
    """
    Returns the median relative prediction error for each node, over all
    workflows and tasks.
    """
    print("node_error was called with: ", args)
    results = run_experiment(
        resource_x=args.resource_x,
        resource_y=args.resource_y,
        scale_bayesian_model=args.scale_bayesian_model,
        scale_median_model=args.scale_median_model,
        experiment_number=args.experiment_number)
    # TODO row is misleading, this is a dataframe?

    def median_error(row):
        return np.median(np.abs(row["y"] - row["yhat"]) / row["yhat"])
    median_errors = results.groupby("node").apply(median_error)
    print(median_errors)


@register(registered_scripts)
@option('-e', '--experiment-number', nargs='+', default=['1', '2'])
@option('-o', '--output-file', default='-')
@analysis
def results_csv(args):
    """
    Writes the predictions for all workflows, tasks, nodes, and given
    experiment numbers to a given file or stdout. If '-o' is not specified or
    '-o -' is given the predictions are written to stdout. In any other case
    the argument to '-o' is treated as the path of the file to write to. This
    script refuses to overwrite existing files.

    Output format is as follows:

    workflow;task;node;x;y
    """
    out = ""
    for i in args.experiment_number:
        results = run_experiment(experiment_number=str(i))
        results["workflow"] = results["workflow"].apply(lambda s: s.lower())
        results["task"] = results["task"].apply(lambda s: s.lower())
        results["node"] = results["node"].apply(lambda s: s.lower())
        results["x"] = results["x"].apply(lambda x: int(x))
        results["y"] = results["y"].apply(lambda y: int(y))
        out += results.to_csv(sep=";", columns=["workflow", "task", "node", "x", "y"],
                              header=False, index=False)
    if args.output_file == "-":
        print(out)
        return
    if os.path.isfile(args.output_file):
        print("refusing to overwrite existing file", file=sys.stderr)
        exit(-1)
    with open(args.output_file, "w") as file:
        file.write(out)


@register(registered_scripts)
@option('-s', '--save')
@option("--scale-bayesian-model",  type=toBool, default=True)
@option("--scale-median-model", type=toBool, default=False)
@option('-x', '--resource-x', default="TaskInputSizeUncompressed")
@option('-y', '--resource-y', default="Realtime")
@analysis
def workflow_node_error(args):
    """
    Visualizes the distribution of relative absolute error over all traces
    nodes and workflows.

    Creates one figure for each workflow. Each figure shows boxplots for each
    node and each boxplot shows the distribution of the relative absolute error
    over all the traces of the given workflow that ran on the given node.
    """
    results = run_experiment(
        resource_x=args.resource_x,
        resource_y=args.resource_y,
        scale_bayesian_model=args.scale_bayesian_model,
        scale_median_model=args.scale_median_model)
    workflows = results["workflow"].unique()
    nodes = results["node"].unique()
    # add column with relative absolute error
    results["rae"] = results.apply(
        lambda row: np.abs(row["y"] - row["yhat"]) / row["yhat"], axis=1)
    # transform into data structure that can be plotted
    data = {}
    for workflow in workflows:
        data[workflow] = []
        for node in nodes:
            data[workflow].append(
                results.loc[(results["workflow"] == workflow) & (results["node"] == node), "rae"])

    fig, axs = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
    axs = axs.flatten()
    for i in range(len(workflows)):
        axs[i].set_title(workflows[i])
        axs[i].boxplot(data[workflows[i]])
        axs[i].set_xticklabels(nodes, rotation=-45, ha='left')
    axs[0].set_yscale("log")
    axs[0].set_ylim(bottom=results.loc[results["rae"] > 0, "rae"].min())
    axs[0].set_ylabel("relative absolute prediction error")
    axs[2].set_xlabel("nodes")
    fig.tight_layout()
    if args.save != "":
        plt.savefig(args.save)
    else:
        plt.show()


@ register(registered_scripts)
@ option("-w", "--workflow", default="eager")
@ option("--scale-bayesian-model",  type=toBool, default=True)
@ option("--scale-median-model", type=toBool, default=False)
@ option('-x', '--resource-x', default="TaskInputSizeUncompressed")
@ option('-y', '--resource-y', default="Realtime")
@ analysis
def node_task_error(args):
    """
    Creates a plot showing the average relative error for each task and node for
    the given workflow.
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


@ register(registered_scripts)
@ option("--scale", choices=["log", "linear"], default="log")
@ analysis
def scale_median_model(args):
    '''
    Answers the question if lotaru should scale the outputs of its median models.
    Shows the distribution of errors of predictions made with scaled median models
    and unscaled median models.
    '''
    results_scaled = run_experiment(scale_median_model=True)
    results_unscaled = run_experiment(scale_median_model=False)

    def get_errors(df):
        return df[df["model"] == MedianModel].apply(lambda row: np.abs(row["y"] - row["yhat"]) / row["yhat"], axis=1)
    errors_scaled = get_errors(results_scaled)
    errors_unscaled = get_errors(results_unscaled)
    plt.yscale(args.scale)
    plt.boxplot([list(errors_scaled), list(errors_unscaled)],
                labels=["scaled", "unscaled"])
    plt.show()


@ register(registered_scripts)
@ option("-w", "--workflow", default="eager")
@ option("-e", "--experiment-number", default="1")
@ analysis
def training_traces(args):
    '''
    Let x be the number of instances of a given task during one workflow execution.
    This script prints the unique values of x for all tasks and workflow executions
    in our traces.

    For example if x is [1] that means that during all workflow exections each task
    had only one instance. If x is [1, 2] that means some tasks during some workflow
    executions had two instances while the others only had one.
    '''
    trace_reader = TraceReader(os.path.join("data", "traces"))
    all_data = trace_reader.get_trace(args.workflow.lower(), "local")
    if args.experiment_number == "0":
        all_training_data = all_data[all_data["Label"].apply(
            lambda s: s[:6] == "train-")]
    else:
        all_training_data = all_data[all_data["Label"] == (
            "train-" + args.experiment_number)]
    results = all_training_data.groupby(
        ["Workflow", "WorkflowInputSize", "Task"]).count()
    results.drop(labels=results.columns[1:], axis=1, inplace=True)
    results.columns = ["count"]
    print("unique number of traces per workflow workflow input size and task: ",
          results["count"].unique())
