import os
import sys

from Lotaru import LotaruInstance
from TraceReader import TraceReader

# writes predictions for all workflows, tasks and inputs to csv files
if __name__ == '__main__':
    # setup
    workflows = ["eager", "methylseq", "chipseq", "atacseq", "bacass"]
    nodes = ["asok01", "asok02", "n1", "n2", "c2", "local"]
    experiment_number = sys.argv[1]
    resource_x = "TaskInputSizeUncompressed"
    resource_y = "Realtime"
    scale_bayesian_model = True
    scale_median_model = False
    trace_reader = TraceReader(os.path.join("data", "traces"))

    # create one lotaru instance per workflow
    workflow_lotaru_instance_map = {}
    for workflow in workflows:
        lotaru_instance = LotaruInstance(workflow, experiment_number, resource_x,
                                         resource_y, trace_reader, scale_bayesian_model, scale_median_model)
        lotaru_instance.train_models()
        workflow_lotaru_instance_map[workflow] = lotaru_instance

    # print predictions for all workflows, tasks and nodes
    for workflow in workflows:
        lotaru_instance = workflow_lotaru_instance_map[workflow]
        for task in lotaru_instance.get_tasks():
            for node in nodes:
                test_data = trace_reader.get_test_data(workflow, task, node)
                x = test_data[resource_x].to_numpy()
                y = lotaru_instance.get_prediction(task, node, x.reshape(-1, 1))
                for i in range(x.size):
                    print('{};{};{};{};{};{}'.format(node, workflow,
                                                     task.lower(), experiment_number, int(x[i]), int(y[i])))

