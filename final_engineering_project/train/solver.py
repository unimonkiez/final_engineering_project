from typing import Optional
import torch
import time
from torch.utils.data import DataLoader


class Solver(object):
    def __init__(
        self,
        data: DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        model_path: str,
        save_model_every: Optional[int],
        print_progress_every: Optional[int],
    ) -> None:
        self._data = data
        self._model = model
        self._optimizer = optimizer
        self._criterion = torch.nn.MSELoss(reduction="sum")
        self._model_path = model_path
        self._save_model_every = save_model_every
        self._print_progress_every = print_progress_every

    def train(
        self,
    ) -> None:
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

            if self._save_model_every is not None:
                if iteration % self._save_model_every == 0:
                    torch.save(self._model.state_dict(), self._model_path)

            if self._print_progress_every is not None:
                if iteration % self._print_progress_every == 0:
                    now = time.time()
                    print(
                        "trained {number} of batches, this batch took {diff} seconds.".format(
                            number=iteration,
                            diff=now - previous_time,
                        ),
                    )
                    previous_time = now
        print("Finished training!")
