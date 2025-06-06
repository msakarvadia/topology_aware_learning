import numpy as np
import networkx as nx
from numpy import random
import os

"""
In this file we create the topologies we will use in our softmax aggregation function experiments.

"""


def mk_softmax_topos() -> tuple[list[str], list[list[int]]]:
    # return (paths of topo file names, nodes to test per topo)
    softmax_dir = "softmax_topology"
    os.makedirs(softmax_dir, exist_ok=True)

    # graphs = []
    graphs = {}

    g = nx.barabasi_albert_graph(n=33, m=1, seed=0)
    graphs["barabasi_albert_33_1"] = g

    g = nx.barabasi_albert_graph(n=33, m=2, seed=0)
    graphs["barabasi_albert_33_2"] = g

    g = nx.barabasi_albert_graph(n=33, m=3, seed=0)
    graphs["barabasi_albert_33_3"] = g

    paths = []
    idx = 0
    for graph_name, G in graphs.items():
        topology = nx.to_numpy_array(G)
        path = f"{softmax_dir}/topo_{graph_name}.txt"
        np.savetxt(path, topology, fmt="%d")
        paths.append(path)

    return paths
