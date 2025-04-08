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


def mk_backdoor_topos(num_nodes=5, seed=0) -> tuple[list[str], list[list[int]]]:
    # return (paths of topo file names, nodes to test per topo)
    bd_dir = "bd_topology"
    os.makedirs(bd_dir, exist_ok=True)

    # graphs = []
    graphs = {}

    # WS
    for n in [8, 16, 33]:
        g = nx.connected_watts_strogatz_graph(n=8, k=4, p=0.5, seed=seed)
        graphs[f"ws_{n}_4_05_{seed}"] = g

    # BA
    g = nx.barabasi_albert_graph(n=33, m=1, seed=seed)
    graphs[f"barabasi_albert_33_1_{seed}"] = g

    g = nx.barabasi_albert_graph(n=16, m=2, seed=seed)
    graphs[f"barabasi_albert_16_2_{seed}"] = g

    g = nx.barabasi_albert_graph(n=33, m=2, seed=seed)
    graphs[f"barabasi_albert_33_2_{seed}"] = g

    g = nx.barabasi_albert_graph(n=64, m=2, seed=seed)
    graphs[f"barabasi_albert_64_2_{seed}"] = g

    g = nx.barabasi_albert_graph(n=33, m=3, seed=seed)
    graphs[f"barabasi_albert_33_3_{seed}"] = g

    g = nx.barabasi_albert_graph(n=8, m=2, seed=seed)
    graphs[f"barabasi_albert_8_2_{seed}"] = g

    # SB
    sizes = [11, 11, 11]
    self_conect = 0.5
    community_connect = 0.009
    probs = [
        [self_conect, community_connect, community_connect],
        [community_connect, self_conect, community_connect],
        [
            community_connect,
            community_connect,
            self_conect,
        ],
    ]
    g = nx.stochastic_block_model(sizes, probs, seed=seed)
    graphs[f"sb_11_05_0009_{seed}"] = g

    sizes = [11, 11, 11]
    self_conect = 0.5
    community_connect = 0.05
    probs = [
        [self_conect, community_connect, community_connect],
        [community_connect, self_conect, community_connect],
        [
            community_connect,
            community_connect,
            self_conect,
        ],
    ]
    g = nx.stochastic_block_model(sizes, probs, seed=seed)
    graphs[f"sb_11_05_005_{seed}"] = g

    sizes = [11, 11, 11]
    self_conect = 0.5
    community_connect = 0.09
    probs = [
        [self_conect, community_connect, community_connect],
        [community_connect, self_conect, community_connect],
        [
            community_connect,
            community_connect,
            self_conect,
        ],
    ]
    g = nx.stochastic_block_model(sizes, probs, seed=seed)
    graphs[f"sb_11_05_009_{seed}"] = g

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
