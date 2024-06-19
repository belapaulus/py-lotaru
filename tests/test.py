import unittest
import os
import io
import pandas as pd
import numpy as np
import context
from lotaru.OnlineInstance import KSModel
from lotaru.Constants import NODES
from lotaru.RunExperiment import run_experiment
from lotaru.TraceReader import TraceReader

INDEX = ["workflow", "task", "node", "x"]


class TestLotaru(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tr = TraceReader()

    def test_lotaru(self):
        for estimator in ["online-m", "lotaru-g", "lotaru-a"]:
            for experiment_number in ["1", "2"]:
                print(
                    f"testing {estimator} with experiment number {experiment_number}")
                ref = self.get_java_lotaru_predictions(
                    estimator=estimator, experiment_number=experiment_number)
                results = self.get_py_lotaru_predictions(
                    estimator=estimator, experiment_number=experiment_number)
                err = np.abs(ref["y"] - results["y"]) / ref["y"]
                self.assertTrue(np.all(err < 1e-10))

    def _test_testdata(self):
        ref = self.get_java_lotaru_predictions()
        results = self.get_py_lotaru_predictions()
        self.assertTrue(ref.index.equals(results.index))

    def get_py_lotaru_predictions(self, experiment_number="1", estimator="lotaru-g"):
        df = run_experiment(
            experiment_number=experiment_number, model=estimator)
        df.set_index(INDEX, inplace=True)
        df.drop(["yhat", "rae"], axis=1, inplace=True)
        return df.sort_index()

    def get_java_lotaru_predictions(self, experiment_number="1", estimator="lotaru-g"):
        rename_estimator = {
            "lotaru-g": "lotaru-g",
            "lotaru-a": "lotaru-a",
            "online-m": "onlinem",
            "online-p": "onlinep",
        }
        estimator = rename_estimator[estimator]
        # read all prediction files, for each file skip the header and append
        # the predictions to data, construct dataframe from data
        header = "node,workflow,task,estimator,feature,wfinputsize,y,yhat,deviation"
        data = header
        for node in NODES:
            file_name = os.path.join(
                "java-lotaru-results", f"tasks_lotaru_{node}.csv")
            with open(file_name) as file:
                data += "\n" + "\n".join(file.readlines()[1:])
        df = pd.read_csv(io.StringIO(data.lower()))
        # split workflow and experiment number into seperate columns
        df["experiment_number"] = df["workflow"].apply(
            lambda s: str(int(s[-1]) + 1))
        df["workflow"] = df["workflow"].apply(lambda s: s[:-2])
        # filter for requested experiment_number and estimator
        df = df[df["experiment_number"] == experiment_number]
        df = df[df["estimator"] == estimator]
        # get taskinputsizeuncompressed (x) from wfinputsize, this only works
        # if each task is executed only once per workflow and each workflow
        # is executed only once per wfinputsize
        # TODO add assertion
        test_data = self.tr.get_all_test_data()
        test_data.set_index(
            ["workflow", "task", "machine", "workflowinputuncompressed"], inplace=True)
        df["x"] = df.apply(lambda row: test_data.loc[row["workflow"], row["task"], row["node"], int(
            row["wfinputsize"])]["taskinputsizeuncompressed"], axis=1)
        # set index and remove extra columns
        df.set_index(INDEX, inplace=True)
        df.drop(["feature", "deviation", "experiment_number",
                "estimator", "wfinputsize", "yhat"], axis=1, inplace=True)
        return df.sort_index()


if __name__ == "__main__":
    unittest.main()
