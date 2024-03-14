import os

import numpy as np
import matplotlib.pyplot as plt

from TraceReader import TraceReader
from Lotaru import get_task_model_map
from NodeFactor import get_node_factor_map

"""
creates one figure for each workflow
each figure shows boxplots for each node
each boxplot shows the distribution of the relative absolute error
over all the tasks of the given workflow that ran on the given node
"""
if __name__ == '__main__':
    workflows = ["eager", "methylseq", "chipseq", "atacseq", "bacass"]
    nodes = ["asok01", "asok02", "n1", "n2", "c2", "local", "wally"]
    experiment_number = "1"
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
    plt.figure()
    num_rows = 2
    num_cols = 3
    plt.subplot(num_rows, num_cols, 1)
    for i in range(len(workflows)):
        workflow = workflows[i]
        plt.subplot(num_rows, num_cols, i+1)
        plt.yscale("log")
        plt.title(workflow)

        node_err_map = {}
        for node in nodes:
            node_err_map[node] = []
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
        #print(workflow)
        #for k, v in node_err_map.items():
        #    print(k, len(v))
        plt.boxplot(node_err_map["c2"])

    plt.show()

