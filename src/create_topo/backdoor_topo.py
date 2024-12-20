import numpy as np
import networkx as nx
from numpy import random
import os

"""
In this file we create the topologies we will use in our backdoor experiments.

"""


def mk_backdoor_topos() -> tuple[list[str], list[list[int]]]:
    # return (paths of topo file names, nodes to test per topo)
    bd_dir = "bd_topology"
    os.makedirs(bd_dir, exist_ok=True)

    graphs = []

    G = nx.complete_graph(3)
    graphs.append(G)
    G = nx.complete_graph(10)
    graphs.append(G)

    paths = []
    nodes = []
    for idx, G in enumerate(graphs):
        topology = nx.to_numpy_array(G)
        path = f"{bd_dir}/topo_{idx}.txt"
        np.savetxt(path, topology, fmt="%d")
        paths.append(path)
        nodes.append(
            [0]
        )  # these are the list of nodes for each graph that need to be backdoored

    return paths, nodes
