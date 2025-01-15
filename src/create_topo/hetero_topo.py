import numpy as np
import networkx as nx
from numpy import random
import os
from src.effective_neighbors import get_n_placement_locations


"""
In this file we create the topologies we will use in our heterogeneous data distribution experiments.

"""


def mk_hetero_topos() -> tuple[list[str], list[list[int]]]:
    # return (paths of topo file names, nodes to test per topo)
    hetero_dir = "hetero_topology"
    os.makedirs(hetero_dir, exist_ok=True)

    # graphs = []
    graphs = {}

    g = nx.barabasi_albert_graph(n=100, m=2)
    graphs["barabasi_albert"] = g
    # graphs.append(g)

    g = nx.barabasi_albert_graph(n=66, m=2)
    graphs["barabasi_albert"] = g
    # graphs.append(g)

    g = nx.barabasi_albert_graph(n=33, m=2)
    graphs["barabasi_albert"] = g
    # graphs.append(g)

    paths = []
    idx = 0
    for graph_name, G in graphs.items():
        topology = nx.to_numpy_array(G)
        path = f"{hetero_dir}/topo_{graph_name}.txt"
        np.savetxt(path, topology, fmt="%d")
        paths.append(path)

    return paths
