import os

import numpy as np
import matplotlib.pyplot as plt

from TraceReader import TraceReader
from Lotaru import get_task_model_map
from NodeFactor import get_node_factor_map

"""
returns the median relative prediction error
for each node over all workflows and tasks
"""
if __name__ == '__main__':
    workflows = ["eager", "methylseq", "chipseq", "atacseq", "bacass"]
    nodes = ["asok01", "asok02", "n1", "n2", "c2", "local", "wally"]
    experiment_number = "0"
    resource_x = "TaskInputSizeUncompressed"
    resource_y = "Realtime"
    scale = True
    trace_reader = TraceReader(os.path.join("data", "traces"))
    workflow_task_model_map = {}
    for workflow in workflows:
        training_data = trace_reader.read_training_data(workflow, experiment_number)
        workflow_task_model_map[workflow] = get_task_model_map(resource_x, resource_y, training_data)
    if scale:
        node_factor_map = get_node_factor_map(os.path.join("data", "benchmarks"))
    node_err_map = {}
    for node in nodes:
        node_err_map[node] = []
        for workflow in workflows:
            for task in workflow_task_model_map[workflow].keys():
                test_data = trace_reader.read_test_data(workflow, node)
                task_test_data = test_data[test_data["Task"].apply(lambda s: s == task)]
                x = task_test_data[resource_x].to_numpy()
                yhat = task_test_data[resource_y].to_numpy()
                if scale:
                    factor = node_factor_map[node]
                else:
                    factor = 1
                y = workflow_task_model_map[workflow][task].predict(x.reshape(-1, 1)) * factor
                err = np.abs(y - yhat) / yhat
                node_err_map[node].extend(list(err))

    for node, err in node_err_map.items():
        print(node, ": ", np.median(err))
