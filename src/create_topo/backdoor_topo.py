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


def mk_backdoor_topos() -> tuple[list[str], list[list[int]]]:
    # return (paths of topo file names, nodes to test per topo)
    bd_dir = "bd_topology"
    os.makedirs(bd_dir, exist_ok=True)

    # graphs = []
    graphs = {}

    """
    g = nx.barabasi_albert_graph(n=33, m=1, seed=0)
    graphs["barabasi_albert_33_1"] = g
    """

    g = nx.barabasi_albert_graph(n=33, m=2, seed=0)
    graphs["barabasi_albert_33_2"] = g
    # graphs.append(g)

    """
    g = nx.barabasi_albert_graph(n=66, m=2, seed=0)
    graphs["barabasi_albert_low"] = g

    sizes = [33, 33, 33]
    probs = [[0.4, 0.009, 0.009], [0.009, 0.4, 0.009], [0.009, 0.009, 0.4]]
    g = nx.stochastic_block_model(sizes, probs, seed=0)
    graphs["stochastic_block"] = g

    g = nx.ring_of_cliques(10, 4)
    graphs["ring_clique"] = g

    # graphs.append(g)
    g = nx.barbell_graph(m1=10, m2=3, create_using=None)
    graphs["barbell"] = g

    g = nx.complete_graph(10)
    graphs["complete"] = g

    g = nx.path_graph(10)
    graph['path'] = g

    g = nx.cycle_graph(10)
    graphs["cycle"] = g
    """

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
            bd_placement_nodes = get_placement_locations_by_top_n_degree(G, 5)

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
