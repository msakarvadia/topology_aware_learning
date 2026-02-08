# Experiments

## Set up

```
git clone https://github.com/msakarvadia/topology_aware_learning.git
cd topology_aware_learning
module load frameworks
python3 -m venv new_env --system-site-packages
source new_env/bin/activate
pip install parsl 
```

## MVP

How to run to minimum viable product to reproduce errors (assuming you have an interactive compute allocation and are launch experiments mannually):

```
cd topology_aware_learning
module load frameworks
# I assume you have a functioning venv or conda env based on above instructions
source new_env/bin/active
cd src/experiments/

# For a single experiment to deploy: "inner loop"
python dummy_main.py

# For a string of experiments to be deployed and managed within another parsl instance: "outer loop" manages "inner loop"
python dummy_experiments.py
```

## Using PBS

How to submit experiment to the job scheduler:

 - One experiment at a time (open the *.sh script and modify the project allocation/username info before launching)
```
cd topology_aware_learning/scripts/aurora/io_scaling_scripts

# Flare filesystem
qsub -l select=<NUM_NODES> dummy_experiment_launch_flare.sh

# daos w/ dfuse
qsub -l select=<NUM_NODES> dummy_experiment_launch_wo_intercept_daos.sh

# daos w/ intercept
qsub -l select=<NUM_NODES> dummy_experiment_launch_w_intercept_daos.sh
```

- Scaling runs across {16, 32, 64, 128, 256} node (open `scaling.sh` and comment out the qsub commands that don't correspond to desired filesystem)
```
./scaling.sh
```

## Running a REAL experiment

```
# Follow the above environment setup instructions

python decentralized_main.py
```
