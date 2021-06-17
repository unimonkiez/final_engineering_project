from typing import Any, Tuple
import torch
import torch.nn as nn
from torchaudio.models.conv_tasnet import ConvBlock, MaskGenerator


class Model(nn.Module):
    def __init__(
        self,
        o_vector_length: int,
    ) -> None:
        super().__init__()

        self._N = 256
        self._L = 20
        self._B = 256
        self._H = 512
        self._P = 3
        self._X = 8
        self._R = 4
        self._D = 256

        self._enc_stride = self._L // 2

        self._encoder = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self._N,
            kernel_size=self._L,
            stride=self._enc_stride,
            padding=self._enc_stride,
            bias=False,
        )
        self._decoder = torch.nn.ConvTranspose1d(
            out_channels=1,
            in_channels=self._N,
            kernel_size=self._L,
            stride=self._enc_stride,
            padding=self._enc_stride,
            bias=False,
        )

        self._before_o = ConvBlock(
            io_channels=self._B,
            hidden_channels=self._H,
            kernel_size=self._P,
            padding=1,
            no_residual=True,
        )

        self._mask_generator = MaskGenerator(
            input_dim=self._N,
            num_sources=1,
            kernel_size=self._P,
            num_feats=self._B,
            num_hidden=self._H,
            num_layers=self._X,
            num_stacks=self._R,
        )

        self._embedding = nn.Linear(
            o_vector_length,
            self._D,
        )

    def forward(self, y: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        batch_size = y.shape[0]

        c = self._embedding(o)
        c_row, c_column = c.size()
        c3d = c.reshape(c_row, c_column, 1)

        encoded = self._encoder(y)
        _, skip = self._before_o(encoded)
        H = skip
        Z = H * c3d
        mask = self._mask_generator(Z)
        masked = mask * encoded.unsqueeze(1)
        masked3d = masked.view(batch_size, self._N, -1)
        output = self._decoder(masked3d)

        return output
