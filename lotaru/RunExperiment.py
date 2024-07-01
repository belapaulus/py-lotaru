import pandas as pd
import numpy as np

from lotaru.LotaruInstance import LotaruInstance
from lotaru.OnlineInstance import OnlineInstance
from lotaru.NaiveInstance import NaiveInstance
from lotaru.TraceReader import TraceReader
from lotaru.Scaler import Scaler
from lotaru.Constants import WORKFLOWS, NODES, LOTARU_G_BENCH, LOTARU_A_BENCH

# mock perfect instance


class PerfectModel:
    def __init__(self):
        None


class PerfectInstance:
    def __init__(self):
        self.model = PerfectModel()

    def train_models(self):
        None

    def get_model_for_task(self, task):
        return self.model


def run_experiment(workflows=WORKFLOWS.keys(), nodes=NODES, experiment_number="0",
                   resource_x="taskinputsizeuncompressed",
                   resource_y="realtime", estimator="lotaru-g",
                   estimator_opts={}):
    trace_reader = TraceReader()
    # set up the estimator
    default_estimator_opts = {
        'lotaru-g': {'scale_bayesian_model': True, 'scale_median_model': False},
        'lotaru-a': {'scale_bayesian_model': True, 'scale_median_model': False},
    }
    for opt, value in default_estimator_opts.get(estimator, {}).items():
        if opt not in estimator_opts:
            estimator_opts[opt] = value
    workflow_instance_map = {}
    for workflow in workflows:
        training_data = trace_reader.get_training_data(
            workflow, experiment_number, resource_x, resource_y)
        match estimator:
            case "lotaru-g":
                scaler = Scaler("g", workflow, LOTARU_G_BENCH)
                instance = LotaruInstance(training_data, scaler,
                                          scale_bayesian_model=estimator_opts["scale_bayesian_model"],
                                          scale_median_model=estimator_opts["scale_median_model"])
            case "lotaru-a":
                scaler = Scaler("a", workflow, LOTARU_A_BENCH)
                instance = LotaruInstance(training_data, scaler,
                                          scale_bayesian_model=estimator_opts["scale_bayesian_model"],
                                          scale_median_model=estimator_opts["scale_median_model"])
            case "online-m":
                instance = OnlineInstance(training_data, "m")
            case "online-p":
                instance = OnlineInstance(training_data, "p")
            case 'naive':
                instance = NaiveInstance(training_data)
            case 'perfect':
                instance = PerfectInstance()
            case _:
                print("unknown estimator", file=sys.stderr)
                exit(-1)
        instance.train_models()
        workflow_instance_map[workflow] = instance

    # run the experiment
    results = pd.DataFrame(
        columns=["workflow", "task", "node", "model", "x", "yhat", "y", "rae"])
    for workflow in workflows:
        for task in WORKFLOWS[workflow]:
            for node in nodes:
                instance = workflow_instance_map[workflow]
                model_type = type(instance.get_model_for_task(task))
                test_data = trace_reader.get_test_data(workflow, task, node)
                x = test_data[resource_x].to_numpy().reshape(-1, 1)
                yhat = test_data[resource_y].to_numpy()
                if type(instance) is PerfectInstance:
                    y = yhat
                else:
                    y = instance.get_prediction(task, node, x)
                rae = np.abs((y - yhat) / yhat)
                for i in range(x.size):
                    results.loc[results.index.size] = [
                        workflow, task, node, model_type, x[i][0], yhat[i], y[i], rae[i]]

    return results
