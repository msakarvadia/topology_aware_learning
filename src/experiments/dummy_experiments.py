from __future__ import annotations
import parsl

# from parsl.app.app import python_app
from src.experiments.parsl_setup import get_parsl_config
from src.experiments.parsl_setup import run_dummy_experiment

if __name__ == "__main__":
    ######### Parsl
    config, num_accelerators = get_parsl_config("experiment_per_node")

    parsl.load(config)
    #########

    num_experiments = 200
    futures = [
        run_dummy_experiment(machine_name=args.parsl_executor)
        for i in range(num_experiments)
    ]

    experiment_num = 0
    for future, args in zip(futures, param_list):
        print(f"Waiting for {future}")
        try:
            print(f"Got result {future.result()}")
        except Exception as e:
            print(f"Failing w/ exception: {e}")
            print(f"Details of failed experiment {experiment_num}:")
            print(args)
        experiment_num += 1
