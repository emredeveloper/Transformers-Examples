import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def load_rope_module(unique_name: str):
    base_dir = Path(__file__).resolve().parents[1]
    module_path = base_dir / "Architecture" / "partial-rope-full-rope.py"
    spec = importlib.util.spec_from_file_location(unique_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_partial_rope_preserves_non_rotary_dimensions():
    module = load_rope_module("rope_partial_module")
    dim = 12
    partial_factor = 0.5
    rope = module.PartialRoPE(dim=dim, partial_rotary_factor=partial_factor)

    torch.manual_seed(0)
    q = torch.randn(1, 2, 3, dim)
    k = torch.randn(1, 2, 3, dim)

    q_embed, k_embed = rope(q.clone(), k.clone())

    assert q_embed.shape == q.shape
    assert k_embed.shape == k.shape

    rotary_dim = int(dim * partial_factor)
    assert torch.allclose(q_embed[..., rotary_dim:], q[..., rotary_dim:])
    assert torch.allclose(k_embed[..., rotary_dim:], k[..., rotary_dim:])


def test_attention_with_rope_output_shape():
    module = load_rope_module("rope_attention_module")
    dim = 16
    num_heads = 4
    head_dim = dim // num_heads
    rope = module.FullRoPE(dim=head_dim)
    attention = module.AttentionWithRoPE(dim=dim, num_heads=num_heads, rope_module=rope)

    torch.manual_seed(1)
    dummy_input = torch.randn(2, 5, dim)

    output = attention(dummy_input)

    assert output.shape == dummy_input.shape
    assert torch.isfinite(output).all()
