from __future__ import annotations

from concurrent.futures import as_completed
from parsl.app.app import python_app
import logging
import time
import os
import pandas as pd

APP_LOG_LEVEL = 21
logger = logging.getLogger("decentral_app")


class DummyDecentrallearnApp:
    def __init__(
        self,
        rounds: int = 10,
        num_models: int = 33,
        ckpt_freq: int = 1,
        log_dir: str = "dummy_logs",
    ) -> None:

        logger.log(APP_LOG_LEVEL, f"Initilizing decentral app")
        print("Initializing decentral app")
        self.rounds = rounds
        self.num_models = num_models
        self.log_dir = log_dir

        self.df = pd.DataFrame(
            columns=["round", "ckpt_write_time", "successful_ckpt", "ckpt_size_gb"]
        )
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            self.df.to_csv(f"{self.log_dir}/io_stats.csv", index=False)

        self.ckpt_freq = ckpt_freq

    def run(
        self,
    ) -> (
        list[Result],
        tuple(list[Result], DecentralClient),
        dict[int, dict[int, tuple(list[Result], DecentralClient)]],
    ):
        """Run the application."""
        import torch

        # dummy parsl workflow that runs for 'rounds'
        for round_idx in range(self.rounds):
            print(f"launching tasks for {round_idx=}")
            # launch 'num_models' parsl tasks
            futures = [train(i) for i in range(self.num_models)]
            print("waiting for futures")
            # wait for all tasks
            model_state_dicts = []
            start_time = time.perf_counter()
            for future in as_completed(futures):
                resolved_future = future.result()
                message = resolved_future[0]
                model = resolved_future[1]
                model_state_dicts.append(model.state_dict())
                # print(message)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"Model training took: {elapsed_time:.4f} seconds")

            print(f"Ckpting all models")
            try:
                if round_idx % self.ckpt_freq == 0:
                    ckpt = {
                        "model_state_dicts": model_state_dicts,
                        "round_idx": round_idx,
                    }
                    ckpt_path = f"{self.log_dir}/{round_idx}_ckpt.pth"

                    start_time = time.perf_counter()
                    torch.save(ckpt, ckpt_path)
                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time
                    print(
                        f"I/O operation (network request) took: {elapsed_time:.4f} seconds"
                    )
                    successful_ckpt = os.path.exists(ckpt_path)
                    size_of_file = "n/a"
                    if successful_ckpt:
                        size_of_file = os.path.getsize(ckpt_path) / (1024**3)
                    else:
                        return 1

                    self.df.loc[len(self.df)] = [
                        round_idx,
                        elapsed_time,
                        successful_ckpt,
                        size_of_file,
                    ]
                    self.df.to_csv(f"{self.log_dir}/io_stats.csv", index=False)

            except Exception as e:
                print(e)
                return e

        return 0


@python_app(executors=["decentral_train"])
def train(model_idx):
    # dummy trianing loop
    import torch
    import torchvision.models as models
    import torch.optim as optim

    intel_xpu_count = torch.xpu.device_count()
    if intel_xpu_count > 0:
        import intel_extension_for_pytorch as ipex

    logger.log(APP_LOG_LEVEL, f"Starting Training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("xpu" if torch.xpu.is_available() else device)
    print(f"{device=}")

    num_batches = 1000
    batch_size = 64
    x_dim = y_dim = 28
    fake_training_batch = torch.zeros(batch_size, 3, x_dim, y_dim)

    # Load ResNet18 with default pre-trained ImageNet weights
    model = models.resnet18(weights="DEFAULT")
    model.train()
    model.to(device)

    # initialize training utils for backpropagation
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.MSELoss()

    # aurora specific torch setup
    if intel_xpu_count > 0:
        model, optimizer = ipex.optimize(model, optimizer=optimizer)

    # few rounds of fake training
    for e in range(num_batches):
        output = model(fake_training_batch.to(device))
        fake_labels = output - 1
        loss = criterion(output, fake_labels)  # Calculate loss
        loss.backward()  # Backward pass (compute gradients)
        optimizer.step()

    model.to("cpu")

    return f"trained model = {model_idx=} on {device=}", model
