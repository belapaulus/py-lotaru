import os
import pandas as pd

def get_node_factor_map(benchmark_dir):
    scores = pd.read_csv(os.path.join(benchmark_dir, "benchmarkScores.csv"), index_col=0, dtype={"node": "string"})
    local_cpu = scores.loc["wally", "cpu_score"]
    local_io = scores.loc["wally", "io_score"]
    node_factor_map = {}
    for node in scores.index:
        node_factor_map[node] = ((local_cpu / scores.loc[node, "cpu_score"]) + (
                local_io / scores.loc[node, "io_score"])) / 2
    return node_factor_map

if __name__ == '__main__':
    print(get_node_factor_map("data/benchmarks"))