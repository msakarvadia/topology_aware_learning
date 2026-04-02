import networkx as nx
from src.create_topo.backdoor_topo import get_placement_locations_by_top_n_degree
from src.create_topo.ba_standard_topo import get_ood_node_placements
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import argparse
from pathlib import Path


def get_topo_names(seed=0):
    # topo_names = []
    # topos = []
    property_dicts = []

    for nodes in [33]:
        for topo in [3, 1, 2]:
            topo_name = f"ba_{nodes}_{topo}_{seed}"
            # topo_names.append(topo_name)
            d = {"deg": topo, "name": topo_name, "seed": seed, "nodes": nodes}
            property_dicts.append(d)

    return property_dicts


topo_dicts = get_topo_names()


def get_placements_and_graph(topo_dict):
    topo_name = topo_dict["name"]

    if "ba" in topo_name:
        g = nx.barabasi_albert_graph(
            n=topo_dict["nodes"], m=topo_dict["deg"], seed=topo_dict["seed"]
        )

    # NOTE(MS): in expeirments I actually placed on 6th highest deg...not fourth
    deg_placement_nodes = get_placement_locations_by_top_n_degree(g, 6)
    first_high_deg_node = f"[{deg_placement_nodes[0]},]"
    fourth_high_deg_node = f"[{deg_placement_nodes[3]},]"
    sixth_high_deg_node = f"[{deg_placement_nodes[5]},]"
    print(f"{sixth_high_deg_node=}")

    nodes = []
    num_nodes_placement = [2, 4, 6]
    for num_placements in num_nodes_placement:
        ood_nodes = get_ood_node_placements(g, num_placements, seed)
        nodes.append(ood_nodes)

    nodes.append(first_high_deg_node)
    nodes.append(fourth_high_deg_node)
    nodes.append(sixth_high_deg_node)
    print(f"{nodes=}")
    return nodes, g


parser = argparse.ArgumentParser()
parser.add_argument(
    "--rootdir",
    type=str,
    default="/lus/flare/projects/AuroraGPT/mansisak/distributed_ml/src/experiments/multi_node_ood_logs",
    help="directory path to where all raw experimental results are stored",
)
parser.add_argument(
    "--results_loc",
    type=str,
    default="/lus/flare/projects/AuroraGPT/mansisak/distributed_ml/figs/ood_results",
    help="directory path to where all all compiled results are stored",
)
args = parser.parse_args()

# rootdir = "/lus/flare/projects/AuroraGPT/mansisak/distributed_ml/src/experiments/bd_scheduler_logs"
# results_loc = "/lus/flare/projects/AuroraGPT/mansisak/distributed_ml/figs/results"

num_exp = 0
for data in [
    "camelyon17",
    # "civilcomments",
    # "cifar10",
    # "cifar100",
    "cifar10_vgg",
    "cifar100_vgg",
    "mnist",
    "fmnist",
    "tiny_mem",
]:
    wd = 0
    num_example = 5000
    momentum = 0
    task_type = "multiply"
    ood_types = [
        "bd",
        "weather",
        "blur",
        "noise",
    ]
    scheduler = None
    eta_min = 1
    T_0 = 66
    softmax_coeff = 10
    label_alpha = 1000
    matrix_type = "row_stoch"
    R = 2
    random_data_placement = True
    offset_clients = [
        0,
    ]
    if data == "tiny_mem":
        num_example = 33000
        lr = "0001"
        optimizer = "adam"
        ood_types = [
            "bd",
            "tiny_mem_7",
        ]
    if data == "cifar10":
        lr = "0001"  # 0.0001
        optimizer = "sgd"
    if data == "cifar100":
        lr = "0001"  # 0.0001
        optimizer = "sgd"
    if data == "cifar10_vgg":
        lr = "00001"  # 0.0001
        optimizer = "adam"
        ood_types = ["bd", "frost", "blur", "noise"]
    if data == "cifar100_vgg":
        lr = "00001"  # 0.0001
        optimizer = "adam"
        ood_types = ["bd", "frost", "blur", "noise"]
    if data == "fmnist":
        lr = "001"
        optimizer = "sgd"
    if data == "mnist":
        lr = "001"
        optimizer = "sgd"
    if data == "camelyon17":
        lr = "0001"
        optimizer = "sgd"
        ood_types = [
            "hospital",
        ]
    if data == "civilcomments":
        lr = "00005"
        optimizer = "adamw"
        ood_types = [
            None,
        ]
        label_alpha = 1
        random_data_placement = False
        offset_clients = [
            0,
            1,
            2,
            3,
        ]

    for seed in [0]:
        topo_dicts = get_topo_names(seed=seed)
        for topo_dict in topo_dicts:
            topo_name = topo_dict["name"]
            print(f"{topo_name}, {data}")
            placements, g = get_placements_and_graph(topo_dict)
            if data == "civilcomments":
                # no ood_nodes
                placements = [
                    [
                        0,
                    ],
                ]
            dfs = []
            for placement in placements:
                num_nodes = len(placement)
                if type(placement) == str:
                    # NOTE (To deal w/ single node placement)
                    num_nodes = 1
                node_set = str(placement).replace(" ", "")
                node_set = node_set.replace(",", "_")
                node_set = node_set.replace("[", "")
                node_set = node_set.replace("]", "")
                for offset_client in offset_clients:
                    for epoch in [5, 10, 20]:
                        for many_to_one in [False, True]:
                            for blur_level, noise_level, fog_level in [
                                (1, 1, 1),
                                (3, 5, 5),
                            ]:
                                for agg_strategy in [
                                    "closeCent",
                                    "eigenCent",
                                    "degCent",
                                    "betCent",
                                    "mhCent",
                                    "weighted",
                                    "random",
                                    "unweighted",
                                    "unweighted_fl",
                                    # "degCent_sim",
                                    # "betCent_sim",
                                ]:
                                    # NOTE(MS): different experiments have different proportions (we didn't run all of these combinations)
                                    for ood_proportion in [0.1, 0.02, 0.5]:
                                        ood_proportion_str = str(
                                            ood_proportion
                                        ).replace(".", "")
                                        os.chdir(f"{args.rootdir}")
                                        for ood_type in ood_types:
                                            num_exp += 1
                                            stats_path = f"data_{topo_name}txt_{data}_64_{epoch}_{lr}_False_True_{label_alpha}_1000_10_{seed}_{agg_strategy}_0_None_{ood_type}_{ood_proportion_str}_{node_set}_False_{many_to_one}_{offset_client}_degree_{random_data_placement}_True_5_{momentum}_{softmax_coeff}_{optimizer}_{wd}_09_098_{scheduler}_095_{T_0}_1_{eta_min}_100_1000_{num_example}_16381_20_150_1_{task_type}_evens_{R}_{matrix_type}_001_{blur_level}_{noise_level}_{fog_level}/"
                                            experiment_dir = stats_path
                                            checkpoint_path = f"{stats_path}39_ckpt.pth"  # NOTE(MS): change this back to 39
                                            stats_path = f"{stats_path}client_stats.csv"

                                            exists = os.path.exists(checkpoint_path)
                                            my_dir = Path(stats_path)

                                            if exists:
                                                try:
                                                    client_df = pd.read_csv(stats_path)
                                                except:
                                                    print(
                                                        "error reading stats, continuing"
                                                    )
                                                    print(stats_path)
                                                    continue
                                                client_df["total_epochs"] = (
                                                    client_df.round_idx
                                                    * len(client_df.epoch.unique())
                                                    + client_df.epoch
                                                )
                                                client_df["agg_strategy"] = (
                                                    f"{agg_strategy}"  # _{scheduler}"
                                                )
                                                client_df["softmax_coeff"] = (
                                                    softmax_coeff
                                                )
                                                client_df["ood_node"] = node_set
                                                client_df["ood_proportion"] = (
                                                    ood_proportion
                                                )
                                                client_df["num_ood_nodes"] = num_nodes
                                                client_df["eta_min"] = eta_min
                                                client_df["T_0"] = T_0
                                                client_df["epoch"] = epoch
                                                client_df["label_alpha"] = label_alpha
                                                client_df["ood_type"] = ood_type
                                                client_df["matrix_type"] = matrix_type
                                                client_df["R"] = R
                                                client_df["noise_level"] = noise_level
                                                client_df["blur_level"] = blur_level
                                                client_df["fog_level"] = fog_level
                                                client_df["many_to_one"] = many_to_one
                                                client_df["lr"] = lr
                                                client_df["optimizer"] = optimizer
                                                client_df["random_data_placement"] = (
                                                    random_data_placement
                                                )
                                                if ood_type == None:
                                                    client_df["backdoor_acc"] = 0
                                                # NOTE: try setting below statement to
                                                # if val is None: (not sure if you can equal none)
                                                if scheduler == None:
                                                    scheduler = "None"
                                                client_df["scheduler"] = scheduler
                                                client_df = client_df[
                                                    [
                                                        "total_epochs",
                                                        "agg_strategy",
                                                        "softmax_coeff",
                                                        "ood_node",
                                                        "test_acc",
                                                        "backdoor_acc",
                                                        "ood_type",
                                                        "client_idx",
                                                        "scheduler",
                                                        "T_0",
                                                        "eta_min",
                                                        "epoch",
                                                        "label_alpha",
                                                        "matrix_type",
                                                        "R",
                                                        "num_ood_nodes",
                                                        "ood_proportion",
                                                        "blur_level",
                                                        "noise_level",
                                                        "fog_level",
                                                        "many_to_one",
                                                        "lr",
                                                        "optimizer",
                                                        "random_data_placement",
                                                    ]
                                                ]

                                                client_df = (
                                                    client_df.drop_duplicates()
                                                )  # there is an issue with how I am loading ckpts
                                                dfs.append(client_df.copy())
                                            else:
                                                print(
                                                    "Does not exist:",
                                                    checkpoint_path,
                                                )

            print("SAVING CSV")
            csv_name = f"{topo_name}_{data}_{optimizer}_{lr}_{wd}_{num_example}.csv"
            if not (dfs == []):
                all_client_results = pd.concat(dfs)
                # results.append(all_client_results
                print(csv_name, all_client_results.shape)
                all_client_results.to_csv(f"{args.results_loc}/{csv_name}")
            else:
                print("NO RESULTS: ", csv_name)
            print("-------")

print(f"{num_exp=}")
