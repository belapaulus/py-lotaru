import pandas as pd
import numpy as np

from lotaru.Constants import NODES


class Scaler:
    def __init__(self, scaler_type, workflow, bench_file):
        self.scaler_type = scaler_type
        self.workflow = workflow
        self.bench_file = bench_file
        if scaler_type == "g":
            factors = self._get_lotaru_g_factors()
            self.get_factor = lambda node, task: factors[node]
            return
        if scaler_type == "a":
            factors = self._get_lotaru_a_factors()
            self.get_factor = lambda node, task: factors.loc[(
                node, task), "factor"]
            return
        # TODO error unknown scaler_type

    def _get_lotaru_g_factors(self):
        df = pd.read_csv(self.bench_file, index_col=0,
                         dtype={"node": "string"})
        local_cpu = df.loc["wally", "cpu_score"]
        local_io = df.loc["wally", "io_score"]
        ret = {}
        for node in NODES:
            ret[node] = ((local_cpu / df.loc[node, "cpu_score"]) +
                         (local_io / df.loc[node, "io_score"])) / 2
        return ret

    def _get_lotaru_a_factors(self):
        """
        Calculate the factor for each node and task. Each factor is
        calculated by dividing the score of the given task on the given
        target node by the score of the given task on the local node. The
        score of a node task pair is given by the median of the three values
        in the relevant line of the relevant score file. In case the factor is
        1, we replace it with the median of all factors that are not 1.
        """
        # TODO check file assumptions
        # TODO asok01/2 confusion
        # TODO
        # s.values is a hack to assign the division results back to the
        # data frame by assigning them positionally rather than by index
        # this might lead to problems in cases where the indexes are not
        # aligned but for some reason the assigning by index does not work
        # maybe this information can be helpful:
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#advanced-shown-levels
        df = pd.read_csv(self.bench_file)
        df = df[df["workflow"] == self.workflow]
        df = df.set_index(["node", "task"])
        df = df.drop("workflow", axis=1)
        df["score"] = df.median(axis=1)
        remotes = NODES.copy()
        remotes.remove("local")
        for node in remotes:
            s = df.loc[node, "score"] / df.loc["local", "score"]
            df.loc[node, "factor"] = s.values

        df.loc[df["factor"] == 1, "factor"] = np.NaN
        for node in remotes:
            fill = df.loc[node, "factor"].median()
            df.loc[node, "factor"] = df.loc[node, "factor"].fillna(fill).values
        df.loc["local", "factor"] = 1
        return df
