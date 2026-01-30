from src.dummy_decentralized_app import DummyDecentrallearnApp
from src.experiments.parsl_setup import get_parsl_config
import parsl

if __name__ == "__main__":
    ### Parsl set up
    experiment_config = "aurora_single_experiment"
    config, num_accelerators = get_parsl_config(experiment_config)
    parsl.load(config)

    decentral_app = DummyDecentrallearnApp()

    decentral_app.run()
