from __future__ import annotations
import parsl

# from parsl.app.app import python_app
from src.experiments.parsl_setup import get_parsl_config
from src.experiments.parsl_setup import run_dummy_experiment

import os

# Set a new environment variable
os.environ["TMPDIR"] = "/tmp"

if __name__ == "__main__":
    ######### Parsl
    config, num_accelerators = get_parsl_config("experiment_per_node")
    print(config)
    print(f"{num_accelerators=}")

    parsl.load(config)
    #########

    num_experiments = 2
    futures = [run_dummy_experiment() for i in range(num_experiments)]

    experiment_num = 0
    for future in futures:
        print(f"Waiting for {future}")
        try:
            print(f"Got result {future.result()}")
        except Exception as e:
            print(f"Failing w/ exception: {e}")
        experiment_num += 1
    parsl.dfk().cleanup()
