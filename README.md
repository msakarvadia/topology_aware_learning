# Topology-Aware Knowledge Propagation in Decentralized Learning

Decentralized learning enables collaborative training of models across naturally distributed data without centralized coordination or maintenance of a global model. Instead, devices are organized in arbitrary communication topologies, in which they can only communicate with neighboring devices. Each device maintains its own local model by training on its local data and integrating new knowledge via model aggregation with neighbors. Therefore, knowledge is propagated across the topology via successive aggregation rounds. We study, in particular, the propagation of out-of-distribution (OOD) knowledge. We find that popular decentralized learning algorithms struggle to propagate OOD knowledge effectively to all devices. Further, we find that both the location of OOD data within a topology, and the topology itself, significantly impact OOD knowledge propagation. We then propose topology-aware aggregation strategies to accelerate (OOD) knowledge propagation across devices. These strategies improve OOD data accuracy, compared to topology-unaware baselines, by 123% on average across models in a
topology.

<img width="875" alt="overview figure" src="https://github.com/user-attachments/assets/8c96adde-4e4b-4cde-90ea-a07490497e82" />

This repo is a test Bed for Prototyping Fully-Distributed ML Experiments. The provided expeirmental scripts accompany the paper.



### Note:
**All scripts have been configured/parallelized to run on the [Aurora](https://www.anl.gov/aurora) supercomputer. Aurora has Intel GPUs. This code has only been tested on Intel GPUs. We have built in untested support for running on nodes w/ Nvidia GPU's. We use the [`parsl`](https://parsl.readthedocs.io/en/stable/index.html) Python parallelization framework. Thereofore, to run on your machine, you must first set up a Parsl `config` in [`parsl_setup.py`](https://github.com/msakarvadia/distributed_ml/blob/main/src/experiments/parsl_setup.py).**

## Generate Topologies

- [`src/create_topo`](https://github.com/msakarvadia/distributed_ml/tree/main/src/create_topo) directory for all topology creation scripts (several scripts for generating differen types of topologies)
- [`src/create_topo/create_topologies.py`](https://github.com/msakarvadia/distributed_ml/blob/main/src/create_topo/create_topologies.py) example topologies
- [`src/create_topo/backdoor_topo.py`](https://github.com/msakarvadia/distributed_ml/blob/main/src/create_topo/backdoor_topo.py) All topologies that we used in official paper experiments

## Simple Decentralized learning demo

- [`src/experiments/decentralized_main.py`](https://github.com/msakarvadia/distributed_ml/blob/main/src/experiments/decentralized_main.py) script to run a single configurable decentralized trianing experiment for a single topology
How to configure Parsl:
  - We provide a tested [example parsl config](https://github.com/msakarvadia/distributed_ml/blob/main/src/experiments/parsl_setup.py#L168) for parallelizing your workflow across a single [Aurora](https://www.alcf.anl.gov/aurora) node.
  - We provide an untested [example parsl config](https://github.com/msakarvadia/distributed_ml/blob/main/src/experiments/parsl_setup.py#L194) for parallelizing your workflow across a single [Polaris](https://www.alcf.anl.gov/polaris) node.
How to run:
```
# set up and activate python environment
# first configure your parsl config in parsl_setup.py
python ../create_topo/create_topologies.py # create and save some topologies
python decentralized_main.py --help # to see all argument options
python decentralized_main.py # to run w/ default args on Aurora (for polaris set this following flag: --parsl_executor polaris_experiment_per_node)
```

## Run All Paper Experiments

- [`src/experiments/bd_scheduler.py`](https://github.com/msakarvadia/distributed_ml/blob/main/src/experiments/bd_scheduler.py) script runs every single experiment in the paper
  - This code relies on 2 levels of parallelization:
    - Parallelization within experiments:
      - We provide a tested [example parsl config](https://github.com/msakarvadia/distributed_ml/blob/main/src/experiments/parsl_setup.py#L168) for parallelizing individual experiments across a single [Aurora](https://www.alcf.anl.gov/aurora) node.
      - We provide an untested [example parsl config](https://github.com/msakarvadia/distributed_ml/blob/main/src/experiments/parsl_setup.py#L194) for parallelizing individual experiments across a single [Polaris](https://www.alcf.anl.gov/polaris) node.
    - Parallelization across experiments:
      - We provide a tested [example parsl config](https://github.com/msakarvadia/distributed_ml/blob/main/src/experiments/parsl_setup.py#L144) for parallelizing your workflow across multiple [Aurora](https://www.alcf.anl.gov/aurora) nodes.
      - We provide an untested [example parsl config](https://github.com/msakarvadia/distributed_ml/blob/main/src/experiments/parsl_setup.py#L119) for parallelizing your workflow across multiple [Polaris](https://www.alcf.anl.gov/polaris) nodes.
How to run:
```
# first configure your parsl config in parsl_setup.py
# set up and activate python environment
# first configure your parsl config in parsl_setup.py
python bd_scheduler.py --rounds 40 
```

## Installation

Requirements:
- `python >=3.7,<3.11`
```bash
git clone https://github.com/msakarvadia/distributed_ml.git
cd distributed_ml
conda create -p env python==3.10
conda activate env
pip install -r requirements.txt
pip install -e .
```
### Setting Up Pre-Commit Hooks (for nice code formatting)

#### Black

To maintain consistent formatting, we take advantage of `black` via pre-commit hooks.
There will need to be some user-side configuration. Namely, the following steps:

1. Install black via `pip install black` (included in `requirements.txt`).
2. Install `pre-commit` via `pip install pre-commit` (included in `requirements.txt`).
3. Run `pre-commit install` to setup the pre-commit hooks.

Once these steps are done, you just need to add files to be committed and pushed and the hook will reformat any Python file that does not meet Black's expectations and remove them from the commit. Just re-commit the changes and it'll be added to the commit before pushing.

### A Simple Example of Decentralized Learning
![2_recall](https://github.com/user-attachments/assets/210ab91c-d411-4b09-bc8e-b86e64d20fc3)

The above animation is an example of a fully-connected topology. Nodes are models and edges are commuincation links between models. Each model is given a subset of the MNIST dataset to train over. We visual the accuracies of the "2" label over training time.

## Citation

Please cite this work as:

```bibtex
...
```
