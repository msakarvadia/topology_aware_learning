from __future__ import annotations

from parsl.app.app import python_app
import logging

APP_LOG_LEVEL = 21
logger = logging.getLogger("decentral_app")


class DummyDecentrallearnApp:

    def __init__(
        self,
        rounds: int = 5,
        num_models: int = 3,
    ) -> None:

        logger.log(APP_LOG_LEVEL, f"Initilizing decentral app")
        print("Initializing decentral app")
        self.rounds = rounds
        self.num_models = num_models

    def run(
        self,
    ) -> (
        list[Result],
        tuple(list[Result], DecentralClient),
        dict[int, dict[int, tuple(list[Result], DecentralClient)]],
    ):
        """Run the application.

        Args:

        Returns:
            List of results from each client after each round.
        """

        # dummy parsl workflow that runs for 'rounds'
        for round_idx in range(self.rounds):
            print(f"launching tasks for {round_idx=}")
            # launch 'num_models' parsl tasks
            futures = [train() for i in range(self.num_models)]
            # wait for all tasks
            for future in futures:
                future.result()

        return 0


@python_app(executors=["decentral_train"])
def train():
    import torch
    import torchvision.models as models
    import torch.optim as optim

    intel_xpu_count = torch.xpu.device_count()
    if intel_xpu_count > 0:
        import intel_extension_for_pytorch as ipex

    logger.log(APP_LOG_LEVEL, f"Starting Training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("xpu" if torch.xpu.is_available() else device)

    num_batches = 10
    batch_size = 256
    x_dim = y_dim = 64
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
