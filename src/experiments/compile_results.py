import networkx as nx
from src.create_topo.backdoor_topo import get_placement_locations_by_top_n_degree
import os
import pandas as pd


def get_topo_names():
    # topo_names = []
    # topos = []
    property_dicts = []

    for seed in [0]:
        for nodes in [8, 16, 33]:
            topo_name = f"ws_{nodes}_4_05_{seed}"
            d = {
                "name": topo_name,
                "nodes": nodes,
                "seed": seed,
            }
            property_dicts.append(d)

        for topo in [0.009, 0.05, 0.09]:
            prob = str(topo).replace(".", "")
            topo_name = f"sb_11_05_{prob}_{seed}"
            # topo_names.append(topo_name)
            d = {
                "name": topo_name,
                "prob": topo,
                "seed": seed,
            }
            property_dicts.append(d)
            # topos.append(topo)

        for nodes in [8, 16, 33]:
            for topo in [1, 2, 3]:
                topo_name = f"barabasi_albert_{nodes}_{topo}_{seed}"
                # topo_names.append(topo_name)
                d = {"deg": topo, "name": topo_name, "seed": seed, "nodes": nodes}
                property_dicts.append(d)

        for topo in [64]:
            topo_name = f"barabasi_albert_{topo}_2_{seed}"
            # topo_names.append(topo_name)
            d = {"deg": 2, "name": topo_name, "seed": seed, "nodes": topo}
            property_dicts.append(d)

    return property_dicts


topo_dicts = get_topo_names()


def get_placements_and_graph(topo_dict):
    topo_name = topo_dict["name"]

    if "ws" in topo_name:
        g = nx.connected_watts_strogatz_graph(
            n=topo_dict["nodes"], k=4, p=0.5, seed=topo_dict["seed"]
        )

    if "barabasi" in topo_name:
        g = nx.barabasi_albert_graph(
            n=topo_dict["nodes"], m=topo_dict["deg"], seed=topo_dict["seed"]
        )

    if "sb" in topo_name:
        sizes = [11, 11, 11]
        self_conect = 0.5
        community_connect = topo_dict["prob"]
        probs = [
            [self_conect, community_connect, community_connect],
            [community_connect, self_conect, community_connect],
            [
                community_connect,
                community_connect,
                self_conect,
            ],
        ]
        g = nx.stochastic_block_model(sizes, probs, seed=topo_dict["seed"])

    placement = get_placement_locations_by_top_n_degree(g, n=4)
    return placement, g


rootdir = "/lus/flare/projects/AuroraGPT/mansisak/distributed_ml/src/experiments/bd_scheduler_logs"
results_loc = "/lus/flare/projects/AuroraGPT/mansisak/distributed_ml/figs/results"

for data in [
    "mnist",
    "fmnist",
    "tiny_mem",
    "cifar10_vgg",
    "cifar100_vgg",
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

    # for topo in [1,2,3]:
    topo_dicts = get_topo_names()
    for topo_dict in topo_dicts:
        topo_name = topo_dict["name"]
        print(f"{topo_name}, {data}")
        placement, g = get_placements_and_graph(topo_dict)

        dfs = []
        for backdoor in [
            True,
        ]:  # False
            for x in range(1, len(placement) + 1):
                node = placement[x - 1]  # .item()
                for epoch in [5, 1]:  # 1]:
                    for scheduler in [
                        None,
                        "CA",
                    ]:  # , "exp", "CA"]:
                        for eta_min in [1, 0, -5, -10]:  # 1, -50]:
                            for T_0 in [66, 5, 8, 10]:  # 66,10, 5, 1]:
                                for softmax_coeff in [2, 4, 6, 8, 10, 100, -10]:
                                    for label_alpha in [1000]:  # 1, 10
                                        for agg_strategy in [
                                            "weighted",
                                            "unweighted",
                                            "degCent",
                                            "betCent",
                                            "unweighted_fl",
                                            "random",
                                            "degCent_sim",
                                            "betCent_sim",
                                        ]:
                                            stats_path = f"{rootdir}/data_topo_{topo_name}txt_{data}_64_{epoch}_{lr}_False_True_{label_alpha}_1000_10_0_{agg_strategy}_0_None_{backdoor}_01_{node}_False_True_0_degree_True_True_5_{momentum}_{softmax_coeff}_{optimizer}_{wd}_09_098_{scheduler}_095_{T_0}_1_{eta_min}_100_1000_{num_example}_16381_20_150_1_{task_type}_evens/"
                                            checkpoint_path = f"{stats_path}39_ckpt.pth"
                                            stats_path = f"{stats_path}client_stats.csv"
                                            exists = os.path.exists(checkpoint_path)

                                            experiment_data = tuple(
                                                (
                                                    node,
                                                    scheduler,
                                                    softmax_coeff,
                                                    agg_strategy,
                                                )
                                            )
                                            if exists:
                                                client_df = pd.read_csv(stats_path)
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
                                                client_df["bd_node"] = node
                                                client_df["eta_min"] = eta_min
                                                client_df["T_0"] = T_0
                                                client_df["epoch"] = epoch
                                                client_df["label_alpha"] = label_alpha
                                                client_df["backdoor"] = backdoor
                                                if not backdoor:
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
                                                        "bd_node",
                                                        "test_acc",
                                                        "backdoor_acc",
                                                        "client_idx",
                                                        "scheduler",
                                                        "T_0",
                                                        "eta_min",
                                                        "epoch",
                                                        "backdoor",
                                                        "label_alpha",
                                                    ]
                                                ]

                                                client_df = (
                                                    client_df.drop_duplicates()
                                                )  # there is an issue with how I am loading ckpts
                                                dfs.append(client_df.copy())
                                        # else:
                                        #  print(stats_path)

        print("-------")
        csv_name = f"{topo_name}_{data}_{optimizer}_{lr}_{wd}_{num_example}.csv"
        if not (dfs == []):
            all_client_results = pd.concat(dfs)
            # results.append(all_client_results
            print(csv_name, all_client_results.shape)
            all_client_results.to_csv(f"{results_loc}/{csv_name}")
        else:
            print("NO RESULTS: ", csv_name)
