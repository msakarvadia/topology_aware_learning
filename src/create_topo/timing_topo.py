import numpy as np
import networkx as nx
from numpy import random
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


def mk_timing_topos(num_nodes=5, seed=0) -> tuple[list[str], list[list[int]]]:
    # return (paths of topo file names, nodes to test per topo)
    bd_dir = "timing_topology"
    os.makedirs(bd_dir, exist_ok=True)

    # graphs = []
    graphs = {}

    # BA vary # of nodes
    for n in [8, 16, 33, 64]:
        for m in [3]:
            g = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
            graphs[f"barabasi_albert_{n}_{m}_{seed}"] = g

    paths = []
    nodes = []
    # for idx, G in enumerate(graphs):
    idx = 0
    for graph_name, G in graphs.items():

        if graph_name in ["ring_clique", "barbell"]:
            bd_placement_nodes = get_placement_locations_by_top_n_degree(G, 2)
        if graph_name in ["complete", "cycle"]:
            bd_placement_nodes = [0]
        else:
            bd_placement_nodes = get_placement_locations_by_top_n_degree(G, num_nodes)

        topology = nx.to_numpy_array(G)
        path = f"{bd_dir}/topo_{graph_name}.txt"
        np.savetxt(path, topology, fmt="%d")
        paths.append(path)
        nodes.append(
            bd_placement_nodes
            # [
            #    0,
            # ]
        )  # these are the list of nodes for each graph that need to be backdoored
        idx += 1

    return paths, nodes
