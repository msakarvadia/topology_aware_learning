import numpy as np
import networkx as nx
from numpy import random
import os

"""
In this file we create the topologies we will use in our softmax aggregation function experiments.

"""


def mk_scale_nodes_topos() -> tuple[list[str], list[list[int]]]:
    # return (paths of topo file names, nodes to test per topo)
    scale_nodes_dir = "scale_nodes_topology"
    os.makedirs(scale_nodes_dir, exist_ok=True)

    # graphs = []
    graphs = {}

    g = nx.barabasi_albert_graph(n=16, m=2, seed=0)
    graphs["barabasi_albert_16_2"] = g

    g = nx.barabasi_albert_graph(n=32, m=2, seed=0)
    graphs["barabasi_albert_32_2"] = g

    g = nx.barabasi_albert_graph(n=64, m=3, seed=0)
    graphs["barabasi_albert_64_2"] = g

    paths = []
    idx = 0
    for graph_name, G in graphs.items():
        topology = nx.to_numpy_array(G)
        path = f"{scale_nodes_dir}/topo_{graph_name}.txt"
        np.savetxt(path, topology, fmt="%d")
        paths.append(path)

    return paths
