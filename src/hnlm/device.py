"""Pick the best available device and mixed-precision settings.

The same code runs on a CPU laptop (dev + tests) and on a Colab/Kaggle GPU (real runs).

Mixed precision in one paragraph: GPUs do matrix multiplies much faster in 16-bit floats.
There are two 16-bit formats:
  * bfloat16 has the same exponent range as float32, so values never overflow. Use it when
    the GPU supports it (Ampere and newer: A100, L4, RTX 30xx+).
  * float16 has a tiny range, so very small gradients round to zero ("underflow"). The fix is
    a GradScaler: multiply the loss by a big number before backward(), then divide the
    gradients by it before the optimizer step. Needed on the T4 / P100 GPUs that free
    Colab and Kaggle give you.
On CPU we just use float32: CPU autocast exists but gives little speedup for small models.
"""

from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    amp_dtype: torch.dtype | None  # None means "no autocast, plain float32"

    @property
    def type(self) -> str:
        return self.device.type

    @property
    def needs_grad_scaler(self) -> bool:
        return self.amp_dtype is torch.float16

    def autocast(self):
        """Context manager for the forward pass. A no-op when mixed precision is off."""
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.type, dtype=self.amp_dtype)

    def describe(self) -> str:
        name = self.type
        if self.type == "cuda":
            name = f"cuda ({torch.cuda.get_device_name(self.device)})"
        precision = "float32" if self.amp_dtype is None else str(self.amp_dtype).split(".")[-1]
        return f"{name}, precision={precision}"


def get_device(prefer: str = "auto", mixed_precision: bool = True) -> DeviceInfo:
    """Return the device to train on.

    prefer: "auto" picks cuda > mps > cpu. Pass "cpu" to force CPU (e.g. in tests).
    """
    if prefer == "auto":
        if torch.cuda.is_available():
            prefer = "cuda"
        elif torch.backends.mps.is_available():
            prefer = "mps"
        else:
            prefer = "cpu"
    device = torch.device(prefer)

    amp_dtype = None
    if mixed_precision and device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return DeviceInfo(device=device, amp_dtype=amp_dtype)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
