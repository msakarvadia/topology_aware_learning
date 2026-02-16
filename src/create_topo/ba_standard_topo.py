import numpy as np
import networkx as nx
from numpy import random
import numpy as np
import os
from src.effective_neighbors import get_n_placement_locations
import torch


"""
In this file we create the topologies we will use in our backdoor experiments.

"""


def get_placement_locations_by_top_n_degree(g, n=3):
    deg_cent = nx.degree_centrality(g)

    start = 0
    interval = len(g) // n

    degrees = torch.tensor(list(deg_cent.values()))
    val, ind = torch.sort(degrees, descending=True)

    l = list(range(0, n))

    placement_neighbors = torch.index_select(ind, 0, torch.tensor(l))
    return placement_neighbors.tolist()


def get_ood_node_placements(graph, n, seed):
    """
    this function will give you a list of n random nodes for your graph on which to place OOD data.

    graph --> network x graph
    n --> number of OOD placement nodes you want in graph
    """
    rng = np.random.default_rng(seed)
    unique_randoms = rng.choice(len(graph), size=n, replace=False)

    return unique_randoms.tolist()


def mk_ba_topos(
    num_nodes=5, seed=0, placement_type="multi"
) -> tuple[list[str], list[list[int]]]:
    # return (paths of topo file names, nodes to test per topo)
    # placement type == "multi" or single
    bd_dir = "bd_topology"
    os.makedirs(bd_dir, exist_ok=True)

    # graphs = []
    graphs = {}

    # BA
    for n in [
        33,
    ]:
        for m in [1, 2, 3]:
            g = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
            graphs[f"barabasi_albert_{n}_{m}_{seed}"] = g

    paths = []
    nodes = []
    for graph_name, G in graphs.items():

        num_nodes = [2, 4, 6]
        if placement_type == "degree":
            num_nodes = [
                num_nodes,
            ]
        for num_placements in num_nodes:
            if placement_type == "degree":
                ood_nodes = get_placement_locations_by_top_n_degree(G, num_placements)
            if placement_type == "multi":
                ood_nodes = get_ood_node_placements(G, num_placements, seed)

            topology = nx.to_numpy_array(G)
            path = f"{bd_dir}/topo_{graph_name}.txt"
            np.savetxt(path, topology, fmt="%d")
            paths.append(path)
            nodes.append(
                ood_nodes
            )  # these are the list of nodes for each graph that need to be backdoored

    return paths, nodes
