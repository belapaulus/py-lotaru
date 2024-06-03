import os

import matplotlib.pyplot as plt

from lotaru.analysis.analysis_script import register, option, analysis
from lotaru.TraceReader import TraceReader
from lotaru.Constants import NODES, WORKFLOWS

registered_scripts = []


@register(registered_scripts)
@option("-w", "--workflow", default="eager")
@option("-x", default="TaskInputSizeUncompressed")
@option("-y", default="Realtime")
@analysis
def show_correlation(args):
    '''
    Visualize the relationship between two features in our traces for a given
    workflow. Creates on scatter plot per task.
    '''
    tr = TraceReader(os.path.join("data", "traces"))
    tasks = WORKFLOWS[args.workflow]
    assert (len(tasks) <= 16)

    plt.figure()
    num_rows = 4
    num_cols = 4
    for i in range(len(tasks)):
        task = tasks[i]
        plt.subplot(num_rows, num_cols, i+1)
        plt.title(task)
        for node in NODES:
            data = tr.get_task_data(args.workflow, task, node)
            plt.scatter(data[args.x], data[args.y])
    plt.show()
