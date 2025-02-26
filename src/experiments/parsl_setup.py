from __future__ import annotations

import os
import parsl
from parsl.config import Config

# PBSPro is the right provider for Polaris:
from parsl.providers import PBSProProvider, LocalProvider

# The high throughput executor is for scaling to HPC systems:
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor

# address_by_interface is needed for the HighThroughputExecutor:
from parsl.addresses import address_by_interface

# You can use the MPI launcher, but may want the Gnu Parallel launcher, see below
from parsl.launchers import MpiExecLauncher, GnuParallelLauncher

def get_parsl_config(
        parsl_executor: str = "local",
) -> Config:

    ######### Parsl
    src_dir = "/eagle/projects/argonne_tpc/mansisak/distributed_ml/src/"
    env = "/eagle/projects/argonne_tpc/mansisak/distributed_ml/env/"

    # Get the number of nodes:
    node_file = os.getenv("PBS_NODEFILE")
    with open(node_file,"r") as f:
        node_list = f.readlines()
        num_nodes = len(node_list)

    user_opts = {
        "worker_init": f"module use /soft/modulefiles; module load conda; conda activate {env}; cd {src_dir}",  # load the environment where parsl is installed
        "scheduler_options": "#PBS -l filesystems=home:eagle:grand",  # specify any PBS options here, like filesystems
        "account": "argonne_tpc",
        "queue": "debug",  # e.g.: "prod","debug, "preemptable" (see https://docs.alcf.anl.gov/polaris/running-jobs/)
        "walltime": "00:30:00",
        "nodes_per_block": num_nodes,  # think of a block as one job on polaris, so to run on the main queues, set this >= 10
    }

    aurora_local_provider = LocalProvider(
        # 'num_nodes' debug node
        nodes_per_block=num_nodes,
        launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--ppn 1"),
        init_blocks=1,
        min_blocks=0,
        max_blocks=1,  # Can increase more to have more parallel jobs
    )
    local_provider = LocalProvider(
        # 'num_nodes' debug node
        nodes_per_block=num_nodes,
        init_blocks=1,
        min_blocks=0,
        max_blocks=1,  # Can increase more to have more parallel jobs
    )
    pbs_provider = PBSProProvider(
        launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"),
        account=user_opts["account"],
        queue=user_opts["queue"],
        select_options="ngpus=4",
        # PBS directives (header lines): for array jobs pass '-J' option
        scheduler_options=user_opts["scheduler_options"],
        # Command to be run before starting a worker, such as:
        worker_init=user_opts["worker_init"],
        # number of compute nodes allocated for each block
        nodes_per_block=user_opts["nodes_per_block"],
        init_blocks=1,
        min_blocks=0,
        max_blocks=1,  # Can increase more to have more parallel jobs
        walltime=user_opts["walltime"],
    )
    threadpool_executor = ThreadPoolExecutor(
        label="threadpool_executor",
        max_threads=2,
    )
    if parsl_executor == "aurora_local":
        tile_names = [f'{gid}.{tid}' for gid in range(6) for tid in range(2)]
        executor = HighThroughputExecutor(
            label="decentral_train",
            heartbeat_period=15,
            heartbeat_threshold=120,
            worker_debug=True,
            max_workers_per_node=12,
            available_accelerators=tile_names,
            prefetch_capacity=0,
            provider=aurora_local_provider,
            cpu_affinity="list:0-7,104-111:8-15,112-119:16-23,120-127:24-31,128-135:32-39,136-143:40-47,144-151:52-59,156-163:60-67,164-171:68-75,172-179:76-83,180-187:84-91,188-195:92-99,196-203",
        )
    if parsl_executor == "local":
        executor = HighThroughputExecutor(
            label="decentral_train",
            heartbeat_period=15,
            heartbeat_threshold=120,
            worker_debug=True,
            max_workers_per_node=4,
            available_accelerators=4,
            # available_accelerators=["0", "1", "2", "3"],
            prefetch_capacity=0,
            provider=local_provider,
        )

    if parsl_executor == "node":
        executor = HighThroughputExecutor(
            label="decentral_train",
            heartbeat_period=15,
            heartbeat_threshold=120,
            worker_debug=True,
            max_workers_per_node=4,
            available_accelerators=4,
            # available_accelerators=["0", "1", "2", "3"],
            prefetch_capacity=0,
            provider=pbs_provider,
        )

    config = Config(
        executors=[executor, threadpool_executor],
        checkpoint_mode="task_exit",
        retries=2,
        app_cache=True,
    )

    return config
