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
        for topo in [1, 2, 3]:
            topo_name = f"barabasi_albert_{nodes}_{topo}_{seed}"
            # topo_names.append(topo_name)
            d = {"deg": topo, "name": topo_name, "seed": seed, "nodes": nodes}
            property_dicts.append(d)

    return property_dicts


topo_dicts = get_topo_names()


def get_placements_and_graph(topo_dict):
    topo_name = topo_dict["name"]

    if "barabasi" in topo_name:
        g = nx.barabasi_albert_graph(
            n=topo_dict["nodes"], m=topo_dict["deg"], seed=topo_dict["seed"]
        )

    nodes = []
    num_nodes_placement = [2, 4, 6]
    for num_placements in num_nodes_placement:
        ood_nodes = get_ood_node_placements(g, num_placements, seed)
        nodes.append(ood_nodes)

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
    if data == "tiny_mem":
        num_example = 33000
        lr = "0001"
        optimizer = "adam"
    if data == "cifar10_vgg":
        lr = "00001"  # 0.0001
        optimizer = "adam"
    if data == "cifar10_vgg":
        lr = "00001"  # 0.0001
        optimizer = "adam"
    if data == "fmnist":
        lr = "001"
        optimizer = "sgd"
    if data == "mnist":
        lr = "001"
        optimizer = "sgd"

    for seed in [0]:
        topo_dicts = get_topo_names(seed=seed)
        for topo_dict in topo_dicts:
            topo_name = topo_dict["name"]
            print(f"{topo_name}, {data}")
            placements, g = get_placements_and_graph(topo_dict)

            dfs = []
            for placement in placements:
                num_nodes = len(placement)
                node_set = str(placement).replace(" ", "")
                node_set = node_set.replace(",", "_")
                node_set = node_set.replace("[", "")
                node_set = node_set.replace("]", "")
                epoch = 5
                scheduler = None
                eta_min = 1
                T_0 = 66
                softmax_coeff = 10
                label_alpha = 1000
                matrix_type = "row_stoch"
                R = 2
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
                    os.chdir(f"{args.rootdir}")
                    for ood_type in ["bd", "weather", "blur", "noise"]:
                        # NOTE(MS): TinyMem only has bd OOD data rn
                        if data == "tiny_mem" and (ood_type != "bd"):
                            continue
                        num_exp += 1
                        stats_path = f"data_topo_{topo_name}txt_{data}_64_{epoch}_{lr}_False_True_{label_alpha}_1000_10_{seed}_{agg_strategy}_0_None_{ood_type}_01_{node_set}_False_True_0_degree_True_True_5_{momentum}_{softmax_coeff}_{optimizer}_{wd}_09_098_{scheduler}_095_{T_0}_1_{eta_min}_100_1000_{num_example}_16381_20_150_1_{task_type}_evens_{R}_{matrix_type}_001/"
                        experiment_dir = stats_path
                        checkpoint_path = f"{stats_path}39_ckpt.pth"  # NOTE(MS): change this back to 39
                        stats_path = f"{stats_path}client_stats.csv"

                        exists = os.path.exists(checkpoint_path)
                        my_dir = Path(stats_path)

                        if exists:
                            try:
                                client_df = pd.read_csv(stats_path)
                            except:
                                print("error reading stats, continuing")
                                print(stats_path)
                                continue
                            client_df["total_epochs"] = (
                                client_df.round_idx * len(client_df.epoch.unique())
                                + client_df.epoch
                            )
                            client_df["agg_strategy"] = (
                                f"{agg_strategy}"  # _{scheduler}"
                            )
                            client_df["softmax_coeff"] = softmax_coeff
                            client_df["ood_node"] = node_set
                            client_df["num_ood_nodes"] = len(placements)
                            client_df["eta_min"] = eta_min
                            client_df["T_0"] = T_0
                            client_df["epoch"] = epoch
                            client_df["label_alpha"] = label_alpha
                            client_df["ood_type"] = ood_type
                            client_df["matrix_type"] = matrix_type
                            client_df["R"] = R
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
                                    "ood_type",
                                    "label_alpha",
                                    "matrix_type",
                                    "R",
                                    "num_ood_nodes",
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

            print("-------")
            csv_name = f"{topo_name}_{data}_{optimizer}_{lr}_{wd}_{num_example}.csv"
            if not (dfs == []):
                all_client_results = pd.concat(dfs)
                # results.append(all_client_results
                print(csv_name, all_client_results.shape)
                all_client_results.to_csv(f"{args.results_loc}/{csv_name}")
            else:
                print("NO RESULTS: ", csv_name)

print(f"{num_exp=}")
