from typing import Optional
import torch
import time
from torch.utils.data import DataLoader
from torch import log10, mean

clip_value = 5


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
        self._criterion = torch.nn.MSELoss(reduction="none")
        self._model_path = model_path
        self._save_model_every = save_model_every
        self._print_progress_every = print_progress_every

    def train(
        self,
    ) -> None:
        print("Training..")
        previous_time = time.time()
        previous_loss = 0

        for (i, batch) in enumerate(self._data):
            y = batch["waveform"]
            for event in batch["events"]:
                x = event["waveform"]
                o = event["o_vector"]
                x_pred = self._model(y, o)
                mse_loss = mean(self._criterion(x_pred, x), [1, 2])
                loss = mean(10 * log10(mse_loss))

                self._optimizer.zero_grad()

                loss.backward()
                # torch.nn.utils.clip_grad(self._model.parameters(), clip_value)

                self._optimizer.step()
                previous_loss = loss

            iteration = i + 1

            if self._save_model_every is not None:
                if iteration % self._save_model_every == 0:
                    torch.save(self._model.state_dict(), self._model_path)

            if self._print_progress_every is not None:
                if iteration % self._print_progress_every == 0:
                    now = time.time()
                    print(
                        "trained {number} of batches, loss is {loss}, this batch took {diff} seconds.".format(
                            number=iteration,
                            diff=now - previous_time,
                            loss=previous_loss,
                        ),
                    )
                    previous_time = now
        print("Finished training!")
