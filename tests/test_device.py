import torch

from hnlm.device import get_device


def test_cpu_device_uses_float32():
    info = get_device("cpu")
    assert info.device == torch.device("cpu")
    assert info.amp_dtype is None and not info.needs_grad_scaler
    with info.autocast():
        assert (torch.ones(2) @ torch.ones(2)).dtype == torch.float32


def test_t4_gets_float16_not_emulated_bfloat16(monkeypatch):
    # A T4 is compute capability 7.5: bf16 would only be emulated (slow), so use fp16.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 5))
    info = get_device("cuda")
    assert info.amp_dtype is torch.float16 and info.needs_grad_scaler
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 0))  # A100 / L4
    assert get_device("cuda").amp_dtype is torch.bfloat16
