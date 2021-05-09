import torch
import time
from torch.utils.data import DataLoader

_print_every = 20


class Solver(object):
    def __init__(
        self,
        data: DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self._data = data
        self._model = model
        self._optimizer = optimizer
        self._criterion = torch.nn.MSELoss(reduction="sum")

    def train(self) -> None:
        print("Training..")
        previous_time = time.time()

        for (i, batch) in enumerate(self._data):
            y = batch["waveform"]
            for event in batch["events"]:
                x = event["waveform"]
                o = event["o_vector"]
                x_pred = self._model(y, o)
                loss = self._criterion(x_pred, x)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

            iteration = i + 1
            if iteration % _print_every == 0:
                now = time.time()
                print(
                    "trained {number} of batches, this batch took {diff} seconds.".format(
                        number=iteration,
                        diff=now - previous_time,
                    ),
                )
                previous_time = now
            break
        print("Finished training!")
