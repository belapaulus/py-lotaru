import os
import pandas as pd

from lotaru.Constants import TRACE_DIR, WORKFLOWS, NODES, TRACE_HEADER


class TraceReader:
    def __init__(self, trace_dir=TRACE_DIR):
        self.trace_dir = trace_dir
        # maps from (wf + node) to trace dataframe
        # wf + node might be ambiguous
        # I will have to use nested maps then
        self.wf_node_trace_map = {}

    def get_trace(self, workflow, node):
        if (workflow + node) not in self.wf_node_trace_map:
            path = os.path.join(self.trace_dir, node, workflow, "trace.csv")
            self.wf_node_trace_map[workflow + node] = pd.read_csv(
                str(path), dtype=TRACE_HEADER)
        return self.wf_node_trace_map[workflow + node]

    def get_task_data(self, workflow, task, node):
        all_data = self.get_trace(workflow, node)
        return all_data[all_data["task"] == task]

    def _get_task_training_data(self, local_traces, task, label, resource_x,
                                resource_y):
        task_traces = local_traces[local_traces["task"] == task]
        training_traces = task_traces[task_traces["label"] == label][:6]
        training_data = training_traces[[resource_x, resource_y]]
        training_data.columns = ["x", "y"]
        training_data.reset_index().drop("index", axis=1)
        return training_data

    def get_training_data(self, workflow, experiment_number, resource_x,
                          resource_y):
        '''
        Returns training data for the given workflow.
        The return value is a dictionary with workflow task names as keys and
        dataframes as values. The dataframes contain to columns x and y.

        Training data for a given workflow consists of traces from the local
        machine, that contain the label "train-1" or "train-2". The experiment
        number lets us choose between "train-1" and "train-2". Experiment
        number "0" will return all traces containing either.

        But not all traces with the train-? label qualify as training data.
        Only those, that have among the smallest six workflow input sizes per
        task and experiment number shall be used.
        '''
        assert (experiment_number in ["0", "1", "2"])
        local_traces = self.get_trace(workflow.lower(), "local")
        tasks = local_traces["task"].unique()
        training_data = {}
        for task in tasks:
            task_training_data = pd.DataFrame(
                columns=["x", "y"], dtype="int64")
            if experiment_number in ["0", "1"]:
                df = self._get_task_training_data(
                    local_traces, task, "train-1", resource_x, resource_y)
                task_training_data = pd.concat([task_training_data, df])
            if experiment_number in ["0", "2"]:
                df = self._get_task_training_data(
                    local_traces, task, "train-2", resource_x, resource_y)
                task_training_data = pd.concat([task_training_data, df])
            training_data[task] = task_training_data.reset_index().drop(
                "index", axis=1)
        return training_data

    def get_test_data(self, workflow, task, node):
        all_data = self.get_trace(workflow, node)
        test_data = all_data[all_data["label"] == "test"]
        task_test_data = test_data[test_data["task"] == task]
        return task_test_data

    def get_all_test_data(self):
        df = pd.DataFrame({k: pd.Series(dtype=v)
                          for k, v in TRACE_HEADER.items()})
        for workflow, tasks in WORKFLOWS.items():
            for task in tasks:
                for node in NODES:
                    new = self.get_test_data(workflow, task, node)
                    df = pd.concat([df, new])
        return df
