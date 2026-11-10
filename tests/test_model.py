"""The four core model tests from the plan, plus RoPE / sampling / size checks."""

import pytest
import torch

from hnlm.model.config import SIZES, GPTConfig
from hnlm.model.generate import generate, sample_next
from hnlm.model.gpt import GPT
from hnlm.model.layers import RotaryEmbedding

TINY = dict(vocab_size=97, ctx_len=32, n_layer=2, n_head=2, d_model=32)


def tiny_model(**kw) -> GPT:
    torch.manual_seed(0)
    return GPT(GPTConfig(**{**TINY, **kw})).eval()


@pytest.mark.parametrize("use_sdpa", [True, False])
def test_attention_is_causal(use_sdpa):
    """Changing a FUTURE token must not change the outputs at earlier positions."""
    model = tiny_model(use_sdpa=use_sdpa)
    x = torch.randint(0, 97, (2, 16))
    y = x.clone()
    y[:, 10:] = torch.randint(0, 97, (2, 6))  # change positions 10..15 only
    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(y)
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-5)
    assert not torch.allclose(a[:, 10:], b[:, 10:])  # and later positions DO change


def test_sdpa_matches_manual_attention():
    x = torch.randint(0, 97, (2, 16))
    with torch.no_grad():
        a, _ = tiny_model(use_sdpa=True)(x)
        b, _ = tiny_model(use_sdpa=False)(x)
    assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("use_sdpa", [True, False])
def test_kv_cache_matches_full_forward(use_sdpa):
    """Feeding tokens in chunks through the cache gives the same logits as one full pass."""
    model = tiny_model(use_sdpa=use_sdpa)
    x = torch.randint(0, 97, (2, 20))
    with torch.no_grad():
        full, _ = model(x)
        cache = model.new_kv_cache(2)
        parts = [model(x[:, :7], cache=cache)[0]]  # prefill 7 tokens
        parts += [model(x[:, i : i + 1], cache=cache)[0] for i in range(7, 12)]  # 1 at a time
        parts.append(model(x[:, 12:], cache=cache)[0])  # a chunk of 8 at once
    assert torch.allclose(full, torch.cat(parts, dim=1), atol=1e-5)


def test_generate_same_with_and_without_cache():
    model = tiny_model()
    prompt = torch.randint(0, 97, (1, 5))
    a = generate(model, prompt, 20, temperature=0, use_cache=True)
    b = generate(model, prompt, 20, temperature=0, use_cache=False)
    assert torch.equal(a, b) and a.shape == (1, 25)


def test_overfit_one_batch():
    """Sanity check: a model must be able to memorize a single batch. If loss won't go to ~0,
    something in the model, loss or optimizer wiring is broken."""
    torch.manual_seed(0)
    model = GPT(GPTConfig(**TINY)).train()
    x = torch.randint(0, 97, (4, 24))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        _, loss = model(x[:, :-1], targets=x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05, loss.item()


def test_rope_depends_only_on_relative_position():
    rope = RotaryEmbedding(head_dim=16, max_len=64)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)

    def score(m: int, n: int) -> float:
        return (rope(q, m) * rope(k, n)).sum().item()

    assert score(3, 1) == pytest.approx(score(40, 38), abs=1e-4)  # both distance 2
    assert rope(q, 7).norm().item() == pytest.approx(q.norm().item(), abs=1e-5)  # pure rotation


def test_sampling_controls():
    logits = torch.tensor([[1.0, 5.0, 2.0, 4.0]])
    assert sample_next(logits, temperature=0).item() == 1
    assert sample_next(logits, top_k=1).item() == 1
    g = torch.Generator().manual_seed(0)
    picks = {sample_next(logits, top_k=2, generator=g).item() for _ in range(200)}
    assert picks == {1, 3}  # only the two best can appear
    picks = {sample_next(logits, top_p=0.5, generator=g).item() for _ in range(200)}
    assert picks == {1}  # token 1 alone already has > 50% probability


def test_weight_tying_and_model_sizes():
    model = tiny_model()
    assert model.lm_head.weight is model.tok_emb.weight
    expected = {"5m": (4e6, 6e6), "15m": (13e6, 16e6), "30m": (29e6, 34e6)}
    for name, shape in SIZES.items():
        n = GPT(GPTConfig(vocab_size=16384, **shape)).num_params()
        lo, hi = expected[name]
        assert lo < n < hi, (name, n)
