from typing import Any
import torch
import torch.nn as nn
from torchaudio.models.conv_tasnet import ConvBlock


class Model(nn.Module):
    def __init__(
        self,
        o_vector_length: int,
        device: Any = None,
    ) -> None:
        super().__init__()

        N = 256
        L = 20
        B = 256
        H = 512
        P = 3
        X = 8
        R = 4
        D = 256

        self._encoder = torch.nn.Conv1d(
            in_channels=1,
            out_channels=N,
            kernel_size=L,
            stride=L // 2,
            padding=L // 2,
            bias=False,
        )
        self._decoder = torch.nn.ConvTranspose1d(
            out_channels=1,
            in_channels=N,
            kernel_size=L,
            stride=L // 2,
            padding=L // 2,
            bias=False,
        )

        self._before_o = ConvBlock(
            io_channels=B,
            hidden_channels=H,
            kernel_size=P,
            padding=1,
        )

        self._after_o = [
            ConvBlock(
                io_channels=B,
                hidden_channels=H,
                kernel_size=P,
                padding=1,
            )
            for i in range(3)
        ]

        self._embedding = nn.Linear(
            o_vector_length,
            D,
        )

        if device is not None:
            self._encoder = self._encoder.to(device)
            self._decoder = self._decoder.to(device)
            self._before_o = self._before_o.to(device)
            self._after_o = [x.to(device) for x in self._after_o]
            self._embedding = self._embedding.to(device)

    def forward(self, y: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        c = self._embedding(o)
        c_row, c_column = c.size()
        c3d = c.reshape(c_row, c_column, 1)

        encoded = self._encoder(y)
        residual, skip = self._before_o(encoded)
        H = skip
        Z = H * c3d

        to_decode = 0.0
        for block in self._after_o:
            residual, skip = block(Z)
            if residual is not None:  # the last conv layer does not produce residual
                Z = Z + residual
            to_decode = to_decode + skip

        output = self._decoder(to_decode)

        return output
