from typing import Any, Tuple
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

        self._after_o = [
            ConvBlock(
                io_channels=self._B,
                hidden_channels=self._H,
                kernel_size=self._P,
                padding=1,
                no_residual=i == 2,
            )
            for i in range(3)
        ]

        self._embedding = nn.Linear(
            o_vector_length,
            self._D,
        )

        if device is not None:
            self._encoder = self._encoder.to(device)
            self._decoder = self._decoder.to(device)
            self._before_o = self._before_o.to(device)
            self._after_o = [x.to(device) for x in self._after_o]
            self._embedding = self._embedding.to(device)

    # def _align_num_frames_with_strides(
    #     self, input: torch.Tensor
    # ) -> Tuple[torch.Tensor, int]:
    #     batch_size, num_channels, num_frames = input.shape
    #     is_odd = self._L % 2
    #     num_strides = (num_frames - is_odd) // self._enc_stride
    #     num_remainings = num_frames - (is_odd + num_strides * self._enc_stride)
    #     if num_remainings == 0:
    #         return input, 0

    #     num_paddings = self._enc_stride - num_remainings
    #     pad = torch.zeros(
    #         batch_size,
    #         num_channels,
    #         num_paddings,
    #         dtype=input.dtype,
    #         device=input.device,
    #     )
    #     return torch.cat([input, pad], 2), num_paddings

    def _get_mask(self, input: torch.Tensor) -> torch.Tensor:
        batch_size = input.shape[0]
        mask = 0.0
        for block in self._after_o:
            residual, skip = block(input)
            if residual is not None:  # the last conv layer does not produce residual
                input = input + residual
            mask = mask + skip

        mask = mask.view(batch_size, 1, self._N, -1)

        return mask

    def forward(self, y: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        batch_size = y.shape[0]
        # y_padded, num_pads = self._align_num_frames_with_strides(y)

        c = self._embedding(o)
        c_row, c_column = c.size()
        c3d = c.reshape(c_row, c_column, 1)

        encoded = self._encoder(y)
        _, skip = self._before_o(encoded)
        H = skip
        Z = H * c3d
        mask = self._get_mask(Z)
        masked = mask * encoded.unsqueeze(1)
        masked3d = masked.view(batch_size, self._N, -1)
        output = self._decoder(masked3d)

        return output
