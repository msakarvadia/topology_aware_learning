from __future__ import annotations
import parsl

from parsl.app.app import python_app
from src.experiments.parsl_setup import get_parsl_config

import os

# Set a new environment variable
os.environ["TMPDIR"] = "/tmp"


@python_app(executors=["experiment"])
def run_dummy_experiment(machine_name="aurora", **kwargs):
    from src.dummy_decentralized_app import DummyDecentrallearnApp

    ### Parsl set up
    import parsl
    from src.experiments.parsl_setup import get_parsl_config

    experiment_config = "aurora_single_experiment"
    if "polaris" in machine_name:
        experiment_config = "polaris_single_experiment"
    config, num_accelerators = get_parsl_config(experiment_config)
    try:
        # might have error loading config if parsl
        # session from prior experiment isn't killed properly
        parsl.load(config)
        print("decentral_train parsl config loaded")
    except:
        print("parsl config already loaded")
        # return 1
    ### Parsl set up

    decentral_app = DummyDecentrallearnApp(
        num_models=kwargs["num_models"],
        log_dir=kwargs["log_dir"],
    )
    # NOTE(MS): this is my attempt to handle run failures
    # And to ensure parsl cleans up even if app doesn't successfully run
    try:
        exit_value = decentral_app.run()
    except Exception as e:
        print(e)
        exit_value = 1
    return exit_value


if __name__ == "__main__":
    ######### Parsl
    config, num_accelerators = get_parsl_config("experiment_per_node")
    print(config)
    print(f"{num_accelerators=}")

    parsl.load(config)
    #########

    num_experiments = 2
    num_models_per_experiment = 3
    futures = [
        run_dummy_experiment(
            num_models=num_models_per_experiment, log_dir=f"dummy_logs/exp_{i}"
        )
        for i in range(num_experiments)
    ]

    experiment_num = 0
    for future in futures:
        print(f"Waiting for {future}")
        try:
            print(f"Got result {future.result()}")
        except Exception as e:
            print(f"Failing w/ exception: {e}")
        experiment_num += 1
    parsl.dfk().cleanup()
