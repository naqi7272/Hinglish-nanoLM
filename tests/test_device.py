import torch

from hnlm.device import get_device


def test_cpu_device_uses_float32():
    info = get_device("cpu")
    assert info.device == torch.device("cpu")
    assert info.amp_dtype is None and not info.needs_grad_scaler
    with info.autocast():
        assert (torch.ones(2) @ torch.ones(2)).dtype == torch.float32
