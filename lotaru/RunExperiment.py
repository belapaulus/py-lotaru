import pandas as pd

from joblib import Memory

from lotaru.LotaruInstance import LotaruInstance
from lotaru.TraceReader import TraceReader
from lotaru.Scaler import Scaler
from lotaru.Constants import WORKFLOWS, NODES, LOTARU_G_BENCH

memory = Memory(".cache")


@memory.cache
def run_experiment(workflows=WORKFLOWS, nodes=NODES, experiment_number="0",
                   resource_x="taskinputsizeuncompressed",
                   resource_y="realtime", scaler_type="g",
                   scaler_bench_file=LOTARU_G_BENCH, scale_bayesian_model=True,
                   scale_median_model=False):
    trace_reader = TraceReader()

    # create one lotaru instance per workflow
    workflow_lotaru_instance_map = {}
    for workflow in workflows:
        training_data = trace_reader.get_training_data(
            workflow, experiment_number, resource_x, resource_y)
        scaler = Scaler(scaler_type, workflow, scaler_bench_file)
        li = LotaruInstance(training_data, scaler,
                            scale_bayesian_model=scale_bayesian_model,
                            scale_median_model=scale_median_model)
        li.train_models()
        workflow_lotaru_instance_map[workflow] = li

    results = pd.DataFrame(
        columns=["workflow", "task", "node", "model", "x", "yhat", "y"])
    # print predictions for all workflows, tasks and nodes
    for workflow in workflows:
        lotaru_instance = workflow_lotaru_instance_map[workflow]
        for task in lotaru_instance.tasks:
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
