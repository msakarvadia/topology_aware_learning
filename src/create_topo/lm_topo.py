import numpy as np
import networkx as nx
from numpy import random
import os

"""
In this file we create the topologies we will use in our softmax aggregation function experiments.

"""


def mk_lm_topos() -> tuple[list[str], list[list[int]]]:
    # return (paths of topo file names, nodes to test per topo)
    lm_dir = "lm_topology"
    os.makedirs(lm_dir, exist_ok=True)

    # graphs = []
    graphs = {}

    g = nx.barabasi_albert_graph(n=33, m=1)
    graphs["barabasi_albert_low"] = g

    g = nx.barabasi_albert_graph(n=33, m=2)
    graphs["barabasi_albert_med"] = g

    g = nx.barabasi_albert_graph(n=33, m=3)
    graphs["barabasi_albert_high"] = g

    paths = []
    idx = 0
    for graph_name, G in graphs.items():
        topology = nx.to_numpy_array(G)
        path = f"{lm_dir}/topo_{graph_name}.txt"
        np.savetxt(path, topology, fmt="%d")
        paths.append(path)

    return paths
