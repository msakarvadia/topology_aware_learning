# Experiments

How to run to minimum viable product to reproduce errors:

```
cd topology_aware_learning
module load frameworks
# I assume you have a functioning venv or conda env based on requirements in main readme
source env/bin/active
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
