# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the dp4a W4A8 decode kernel (Ampere)."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

CAP = (
    current_platform.get_device_capability().to_int()
    if current_platform.is_cuda()
    else 0
)
requires_ampere = pytest.mark.skipif(
    not (80 <= CAP < 90), reason="dp4a W4A8 targets sm_80..sm_89"
)

GROUP = 128


def _make_weights(n: int, k: int, dtype: torch.dtype, seed: int = 7):
    torch.manual_seed(seed)
    w_s4 = torch.randint(-8, 8, (n, k), dtype=torch.int8, device="cuda")
    scales = (torch.rand(n, k // GROUP, device="cuda") * 0.9 + 0.1).to(dtype)
    return w_s4, scales


def _pack(w_s4: torch.Tensor) -> torch.Tensor:
    from vllm.model_executor.kernels.linear.mixed_precision.dp4a import _pack_w4

    return _pack_w4(w_s4)


@requires_ampere
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n,k", [(128, 256), (512, 1024), (4096, 4096)])
def test_dequant_roundtrip(dtype, n, k):
    w_s4, scales = _make_weights(n, k, dtype)
    packed = _pack(w_s4)
    folded = (scales.float() / 16.0).to(dtype)
    out = torch.empty(n, k, dtype=dtype, device="cuda")
    ops.w4a8_dp4a_dequant(out, packed, folded)
    ref = w_s4.float() * scales.float().repeat_interleave(GROUP, dim=1)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


@requires_ampere
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bs", [1, 2, 4, 8])
@pytest.mark.parametrize("n,k", [(128, 256), (2048, 4096), (4000, 4096)])
def test_gemm_matches_integer_reference(dtype, bs, n, k):
    w_s4, scales = _make_weights(n, k, dtype)
    packed = _pack(w_s4)
    folded = (scales.float() / 16.0).to(dtype)

    x = (torch.rand(bs, k, device="cuda", dtype=dtype) - 0.5)
    x_q, x_s, _ = ops.scaled_int8_quant(x, symmetric=True)

    out = torch.empty(bs, n, dtype=dtype, device="cuda")
    ops.w4a8_dp4a_gemm(out, x_q, x_s.view(-1).float(), packed, folded)

    # the kernel's integer dot is exact; only accumulation order and the
    # output store differ from this reference
    w_deq = w_s4.float() * scales.float().repeat_interleave(GROUP, dim=1)
    ref = (x_q.float() @ w_deq.t()) * x_s.view(-1, 1).float()
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=1e-1)


@requires_ampere
def test_kernel_selection_and_layer_path():
    from vllm.model_executor.kernels.linear import choose_mp_linear_kernel
    from vllm.model_executor.kernels.linear.mixed_precision.dp4a import (
        Dp4aW4A8LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.scalar_type import scalar_types

    cfg = MPLinearLayerConfig(
        full_weight_shape=(4096, 4096),
        partition_weight_shape=(4096, 4096),
        weight_type=scalar_types.int4,
        act_type=torch.int8,
        group_size=GROUP,
        zero_points=False,
        has_g_idx=False,
    )
    assert choose_mp_linear_kernel(cfg) is Dp4aW4A8LinearKernel


@requires_ampere
@pytest.mark.parametrize("bs", [3, 5, 32])
def test_layer_apply_padding_and_fallback(bs):
    """Odd batches pad to the next kernel size; large ones dequantize."""
    import torch.nn as nn

    from vllm.model_executor.kernels.linear.mixed_precision.dp4a import (
        Dp4aW4A8LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.scalar_type import scalar_types

    n, k = 512, 1024
    dtype = torch.float16
    w_s4, scales = _make_weights(n, k, dtype)

    layer = nn.Module()
    layer.register_parameter(
        "weight_packed", nn.Parameter(w_s4, requires_grad=False)
    )
    layer.register_parameter(
        "weight_scale", nn.Parameter(scales, requires_grad=False)
    )
    cfg = MPLinearLayerConfig(
        full_weight_shape=(k, n),
        partition_weight_shape=(k, n),
        weight_type=scalar_types.int4,
        act_type=torch.int8,
        group_size=GROUP,
        zero_points=False,
        has_g_idx=False,
    )
    kernel = Dp4aW4A8LinearKernel(cfg, "weight_packed", "weight_scale")
    kernel.process_weights_after_loading(layer)

    x = (torch.rand(bs, k, device="cuda", dtype=dtype) - 0.5)
    out = kernel.apply_weights(layer, x)

    w_deq = w_s4.float() * scales.float().repeat_interleave(GROUP, dim=1)
    if bs <= 8:
        x_q, x_s, _ = ops.scaled_int8_quant(x, symmetric=True)
        ref = (x_q.float() @ w_deq.t()) * x_s.view(-1, 1).float()
    else:
        ref = x.float() @ w_deq.t()
    torch.testing.assert_close(out.float(), ref, rtol=3e-2, atol=2e-1)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only checks")
def test_w4a16_kernels_reject_int8_activations():
    """A config that asks for int8 activations must never land on a
    weight-only executor (issue #38064)."""
    from vllm.model_executor.kernels.linear.mixed_precision.humming import (
        HummingLinearKernel,
    )
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.scalar_type import scalar_types

    cfg = MPLinearLayerConfig(
        full_weight_shape=(4096, 4096),
        partition_weight_shape=(4096, 4096),
        weight_type=scalar_types.int4,
        act_type=torch.int8,
        group_size=GROUP,
        zero_points=False,
        has_g_idx=False,
    )
    ok, reason = HummingLinearKernel.can_implement(cfg)
    assert not ok
    assert "int8" in (reason or "")


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only checks")
def test_int8_selection_fails_cleanly_off_ampere():
    """On capabilities without an int8-activation kernel the chooser raises,
    which the scheme converts into a warned W4A16 fallback."""
    from vllm.model_executor.kernels.linear import choose_mp_linear_kernel
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.scalar_type import scalar_types

    cfg = MPLinearLayerConfig(
        full_weight_shape=(4096, 4096),
        partition_weight_shape=(4096, 4096),
        weight_type=scalar_types.int4,
        act_type=torch.int8,
        group_size=GROUP,
        zero_points=False,
        has_g_idx=False,
    )
    if 80 <= CAP < 90:
        assert choose_mp_linear_kernel(cfg, compute_capability=CAP) is not None
    else:
        with pytest.raises(ValueError):
            choose_mp_linear_kernel(cfg, compute_capability=CAP)
