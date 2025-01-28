import numpy as np
import networkx as nx
from numpy import random
import os
from src.effective_neighbors import get_n_placement_locations


"""
In this file we create the topologies we will use in our backdoor experiments.

"""


def mk_backdoor_topos() -> tuple[list[str], list[list[int]]]:
    # return (paths of topo file names, nodes to test per topo)
    bd_dir = "bd_topology"
    os.makedirs(bd_dir, exist_ok=True)

    # graphs = []
    graphs = {}

    g = nx.barabasi_albert_graph(n=33, m=2)
    graphs["barabasi_albert_low"] = g
    # graphs.append(g)

    g = nx.barabasi_albert_graph(n=66, m=2)
    graphs["barabasi_albert_low"] = g

    sizes = [33, 33, 33]
    probs = [[0.4, 0.009, 0.009], [0.009, 0.4, 0.009], [0.009, 0.009, 0.4]]
    g = nx.stochastic_block_model(sizes, probs, seed=0)
    graphs["stochastic_block"] = g

    """
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
            bd_placement_nodes = get_n_placement_locations(G, 0.9, 2)
        if graph_name in ["complete", "cycle"]:
            bd_placement_nodes = [0]
        else:
            bd_placement_nodes = get_n_placement_locations(G, 0.9, 5)

        topology = nx.to_numpy_array(G)
        path = f"{bd_dir}/topo_{graph_name}.txt"
        np.savetxt(path, topology, fmt="%d")
        paths.append(path)
        nodes.append(
            # bd_placement_nodes
            [
                0,
            ]
        )  # these are the list of nodes for each graph that need to be backdoored
        idx += 1

    return paths, nodes
