import pandas as pd
import numpy as np

from lotaru.LotaruInstance import LotaruInstance
from lotaru.OnlineInstance import OnlineInstance
from lotaru.TraceReader import TraceReader
from lotaru.Scaler import Scaler
from lotaru.Constants import WORKFLOWS, NODES, LOTARU_G_BENCH, LOTARU_A_BENCH


def run_experiment(workflows=WORKFLOWS.keys(), nodes=NODES, experiment_number="0",
                   resource_x="taskinputsizeuncompressed",
                   resource_y="realtime", estimator="lotaru-g",
                   estimator_opts={}):
    default_estimator_opts = {
        'lotaru-g': {'scale_bayesian_model': True, 'scale_median_model': False},
        'lotaru-a': {'scale_bayesian_model': True, 'scale_median_model': False},
        'online-m': {},
        'online-p': {},
    }
    for opt, value in default_estimator_opts[estimator].items():
        if opt not in estimator_opts:
            estimator_opts[opt] = value
    trace_reader = TraceReader()

    # create one instance per workflow
    workflow_instance_map = {}
    for workflow in workflows:
        training_data = trace_reader.get_training_data(
            workflow, experiment_number, resource_x, resource_y)
        if estimator == "lotaru-g":
            scaler = Scaler("g", workflow, LOTARU_G_BENCH)
            instance = LotaruInstance(training_data, scaler,
                                      scale_bayesian_model=estimator_opts["scale_bayesian_model"],
                                      scale_median_model=estimator_opts["scale_median_model"])
            instance.train_models()
        if estimator == "lotaru-a":
            scaler = Scaler("a", workflow, LOTARU_A_BENCH)
            instance = LotaruInstance(training_data, scaler,
                                      scale_bayesian_model=estimator_opts["scale_bayesian_model"],
                                      scale_median_model=estimator_opts["scale_median_model"])
            instance.train_models()
        if estimator == "online-m":
            instance = OnlineInstance(training_data, "m")
            instance.train_models()
        if estimator == "online-p":
            instance = OnlineInstance(training_data, "p")
            instance.train_models()
        workflow_instance_map[workflow] = instance

    results = pd.DataFrame(
        columns=["workflow", "task", "node", "model", "x", "yhat", "y", "rae"])
    # print predictions for all workflows, tasks and nodes
    # TODO proper decimals with
    # Decimal('7.325').quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
    # .apply(lambda x: int(x))
    for workflow in workflows:
        for task in WORKFLOWS[workflow]:
            for node in nodes:
                lotaru_instance = workflow_instance_map[workflow]
                model_type = type(lotaru_instance.get_model_for_task(task))
                test_data = trace_reader.get_test_data(workflow, task, node)
                x = test_data[resource_x].to_numpy().reshape(-1, 1)
                yhat = test_data[resource_y].to_numpy()
                y = lotaru_instance.get_prediction(task, node, x)
                rae = np.abs((y - yhat) / yhat)
                for i in range(x.size):
                    results.loc[results.index.size] = [
                        workflow, task, node, model_type, x[i][0], yhat[i], y[i], rae[i]]

    return results
