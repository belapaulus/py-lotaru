import os
import pandas as pd


class TraceReader:
    def __init__(self, trace_dir):
        self.trace_dir = trace_dir

    def read_trace(self, workflow, node):
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
        return pd.read_csv(str(path), dtype=dtypes)

    def read_training_data(self, workflow, experiment_number):
        all_data = self.read_trace(workflow.lower(), "local")
        if experiment_number == "0":
            return all_data[all_data["Label"].apply(lambda s: s[:6] == "train-")]
        return all_data[all_data["Label"] == ("train-" + experiment_number)]

    def read_test_data(self, workflow, node):
        all_data = self.read_trace(workflow, node)
        return all_data[all_data["Label"] == "test"]
