import os
import sys
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from lotaru.LotaruInstance import LotaruInstance
from lotaru.TraceReader import TraceReader

# runs an experiment, returns results as pandas dataframe
def run_experiment(
        workflows=["eager", "methylseq", "chipseq", "atacseq", "bacass"],
        nodes=["asok01", "asok02", "n1", "n2", "c2", "local"],
        experiment_number="0",
        resource_x="TaskInputSizeUncompressed",
        resource_y="Realtime",
        scale_bayesian_model=True,
        scale_median_model=False):
    trace_reader = TraceReader(os.path.join("data", "traces"))

    # create one lotaru instance per workflow
    workflow_lotaru_instance_map = {}
    for workflow in workflows:
        training_data = trace_reader.get_training_data(workflow, experiment_number,
                resource_x, resource_y)
        li = LotaruInstance(training_data, scale_bayesian_model, scale_median_model)
        li.train_models()
        workflow_lotaru_instance_map[workflow] = li

    results = pd.DataFrame(columns=["workflow", "task", "node", "model", "x", "yhat", "y"])
    # print predictions for all workflows, tasks and nodes
    for workflow in workflows:
        lotaru_instance = workflow_lotaru_instance_map[workflow]
        for task in lotaru_instance.get_tasks():
            model_type = type(lotaru_instance.get_model_for_task(task))
            for node in nodes:
                test_data = trace_reader.get_test_data(workflow, task, node)
                x = test_data[resource_x].to_numpy()
                yhat = test_data[resource_y].to_numpy()
                y = lotaru_instance.get_prediction(task, node, x.reshape(-1, 1))
                for i in range(x.size):
                    results.loc[results.index.size] = [workflow, task, node, model_type, x[i], yhat[i], y[i]]

    return results

