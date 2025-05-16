# Experiments

Here we store all of the python experiment scripts which can be launch via `python script_name.py <any args>`

You can also lunch via slurm/PBS by looking in the `../scripts` folder for launch scripts on large scale clusters.

For, the launch scripts specific to the Aurora super computer, looking in the `../scripts/aurora` folder.

- [`decentralized_main.py`](https://github.com/msakarvadia/distributed_ml/blob/main/src/experiments/decentralized_main.py): can run individual training runs for specific networks
- [`bd_scheduler.py`](https://github.com/msakarvadia/distributed_ml/blob/main/src/experiments/bd_scheduler.py): all experiments included in paper
- [`time_experiments.py`](https://github.com/msakarvadia/distributed_ml/blob/main/src/experiments/time_experiments.py): time trials for experiments
- [`parsl_setup.py`](https://github.com/msakarvadia/distributed_ml/blob/main/src/experiments/parsl_setup.py): (IMPORTANT) All model training in each expeirment is parallelized across all avalible GPUs on a node. All expeirments are then parallelized across all avalible nodes. You must set up your parsl configs for your machine in order to run experiments. This script has sample configs for both the Aurora supercomputer (tested), Polaris supercomputer (untested)
- [`compile_results.py`](https://github.com/msakarvadia/distributed_ml/blob/main/src/experiments/compile_results.py): once your experiments have run, this is how you compile all results
