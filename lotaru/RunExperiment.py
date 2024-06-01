import pandas as pd

from joblib import Memory

from lotaru.LotaruInstance import LotaruInstance
from lotaru.TraceReader import TraceReader
from lotaru.Constants import WORKFLOWS, NODES

# runs an experiment, returns results as pandas dataframe

memory = Memory(".cache")


@memory.cache
def run_experiment(
        workflows=WORKFLOWS,
        nodes=NODES,
        experiment_number="0",
        resource_x="TaskInputSizeUncompressed",
        resource_y="Realtime",
        scale_bayesian_model=True,
        scale_median_model=False):
    trace_reader = TraceReader()

    # create one lotaru instance per workflow
    workflow_lotaru_instance_map = {}
    for workflow in workflows:
        training_data = trace_reader.get_training_data(workflow, experiment_number,
                                                       resource_x, resource_y)
        li = LotaruInstance(
            training_data, scale_bayesian_model, scale_median_model)
        li.train_models()
        workflow_lotaru_instance_map[workflow] = li

    results = pd.DataFrame(
        columns=["workflow", "task", "node", "model", "x", "yhat", "y"])
    # print predictions for all workflows, tasks and nodes
    for workflow in workflows:
        lotaru_instance = workflow_lotaru_instance_map[workflow]
        for task in lotaru_instance.get_tasks():
            model_type = type(lotaru_instance.get_model_for_task(task))
            for node in nodes:
                test_data = trace_reader.get_test_data(workflow, task, node)
                x = test_data[resource_x].to_numpy()
                yhat = test_data[resource_y].to_numpy()
                y = lotaru_instance.get_prediction(
                    task, node, x.reshape(-1, 1))
                for i in range(x.size):
                    results.loc[results.index.size] = [
                        workflow, task, node, model_type, x[i], yhat[i], y[i]]

    return results
