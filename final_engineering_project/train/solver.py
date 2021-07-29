from typing import Optional
import torch
import time
from torch.utils.data import DataLoader
from torch import log10, mean
from final_engineering_project.properties import model_path, optimizer_path


class Solver(object):
    def __init__(
        self,
        epoch_size: int,
        data: DataLoader,
        model: torch.nn.Module,
        clip_value: float,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.Optimizer,
        save_model_every: Optional[int],
        print_progress_every: Optional[int],
    ) -> None:
        self._epoch_size = epoch_size
        self._data = data
        self._model = model
        self._optimizer = optimizer
        self._criterion = torch.nn.MSELoss(reduction="none")
        self._save_model_every = save_model_every
        self._print_progress_every = print_progress_every
        self._clip_value = clip_value
        self._scheduler = scheduler

    def train(
        self,
    ) -> None:
        print("Training..")
        previous_time = time.time()
        previous_loss = 0
        previous_norm = 0
        total_number = len(self._data)

        for epoch in range(self._epoch_size):
            for (i, batch) in enumerate(self._data):
                y = batch["waveform"]
                for event in batch["events"]:
                    x = event["waveform"]
                    # x = torch.zeros_like(x)  # Test
                    o = event["o_vector"]
                    x_pred = self._model(y, o)
                    mse_loss = mean(self._criterion(x_pred, x), [1, 2])
                    loss = mean(10 * log10(mse_loss))

                    self._optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad.clip_grad_norm(
                        self._model.parameters(),
                        self._clip_value,
                    )

                    self._optimizer.step()

                    previous_loss = loss
                    previous_norm = 0.0
                    for p in self._model.parameters():
                        param_norm = p.grad.data.norm(2)
                        previous_norm += param_norm.item() ** 2
                    previous_norm = previous_norm ** (1.0 / 2)

                iteration = i + 1

                if self._save_model_every is not None:
                    if iteration % self._save_model_every == 0:
                        torch.save(self._optimizer.state_dict(), optimizer_path)
                        torch.save(self._model.state_dict(), model_path)

                if self._print_progress_every is not None:
                    if iteration % self._print_progress_every == 0:
                        now = time.time()
                        print(
                            "trained {number}/{total_number} of batches, epoch {epoch}/{total_epoch}, loss is {loss}, norm is {norm}, lr is {lr}, this batch took {diff} seconds.".format(
                                number=iteration,
                                total_number=total_number,
                                epoch=epoch + 1,
                                total_epoch=self._epoch_size,
                                diff=now - previous_time,
                                loss=previous_loss,
                                norm=previous_norm,
                                lr=self._scheduler.get_last_lr()[0],
                            ),
                        )
                        previous_time = now
            self._scheduler.step()
        print("Finished training!")
