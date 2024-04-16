import os
import pandas as pd


class TraceReader:
    def __init__(self, trace_dir):
        self.trace_dir = trace_dir
        # maps from (wf + node) to trace dataframe
        # wf + node might be ambiguous
        # I will have to use nested maps then
        self.wf_node_trace_map = {}

    def get_trace(self, workflow, node):
        if (workflow + node) not in self.wf_node_trace_map:
            path = os.path.join(self.trace_dir, node, workflow, "trace.csv")
            dtypes = {
                "Label": "string",
                "Machine": "string",
                "Workflow": "string",
                # "NumberSequences"
                "Task": "string",
                # "WorkflowInputSize"
                # "Realtime"
                # "%cpu"
                # "rss"
                # "rchar"
                # "wchar"
                # "cpus"
                # "read_bytes"
                # "write_bytes"
                # "vmem"
                # "memory"
                # "peak_rss"
                # "TaskInputSize"
                # "TaskInputSizeUncompressed"
                # "WorkflowInputUncompressed"
            }
            self.wf_node_trace_map[workflow + node] = pd.read_csv(str(path), dtype=dtypes)
        return self.wf_node_trace_map[workflow + node]

    def get_task_data(self, workflow, task, node):
        all_data = self.get_trace(workflow, node)
        return all_data[all_data["Task"] == task]

    # use experiment_number = "0" to return all training data
    def get_training_data(self, workflow, experiment_number):
        all_data = self.get_trace(workflow.lower(), "local")
        if experiment_number == "0":
            training_data = all_data[all_data["Label"].apply(lambda s: s[:6] == "train-")]
        else:
            training_data = all_data[all_data["Label"] == ("train-" + experiment_number)]
        return training_data

    def get_test_data(self, workflow, task, node):
        all_data = self.get_trace(workflow, node)
        test_data = all_data[all_data["Label"] == "test"]
        task_test_data = test_data[test_data["Task"].apply(lambda s: s == task)]
        return task_test_data
