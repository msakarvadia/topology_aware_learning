# Experiments

Set up

```
git clone https://github.com/msakarvadia/topology_aware_learning.git
cd topology_aware_learning
module load frameworks
python3 -m venv new_env --system-site-packages
source new_env/bin/activate
pip install parsl 
```

How to run to minimum viable product to reproduce errors:

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

Running a REAL experiment

```
# Follow the above environment setup instructions

python decentralized_main.py
```
