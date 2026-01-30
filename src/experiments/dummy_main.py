from src.dummy_decentralized_app import DummyDecentrallearnApp
from src.experiments.parsl_setup import get_parsl_config
import parsl

if __name__ == "__main__":
    ### Parsl set up
    def run_dummy_experiment():
        from src.dummy_decentralized_app import DummyDecentrallearnApp

        ### Parsl set up - TODO(MS): make parsl executor name an arg for polaris vs aurora
        import parsl
        from src.experiments.parsl_setup import get_parsl_config

        experiment_config = "aurora_single_experiment"
        config, num_accelerators = get_parsl_config(experiment_config)
        print(config)
        parsl.load(config)

        decentral_app = DummyDecentrallearnApp()
        try:
            exit_value = decentral_app.run()
        except:
            exit_value = 1
        parsl.dfk().cleanup()
        return exit_value

    run_dummy_experiment()
