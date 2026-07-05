# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A8-INT kernel selection must not silently pick a weight-only kernel
for int8-activation configs (issue #38064)."""

import pytest
import torch

from vllm.platforms import current_platform

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only selection semantics"
)


def _int8_cfg():
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.scalar_type import scalar_types

    return MPLinearLayerConfig(
        full_weight_shape=(4096, 4096),
        partition_weight_shape=(4096, 4096),
        weight_type=scalar_types.int4,
        act_type=torch.int8,
        group_size=128,
        zero_points=False,
        has_g_idx=False,
    )


@requires_cuda
def test_humming_rejects_int8_activations():
    from vllm.model_executor.kernels.linear.mixed_precision.humming import (
        HummingLinearKernel,
    )

    ok, reason = HummingLinearKernel.can_implement(_int8_cfg())
    assert not ok
    assert "int8" in (reason or "")


@requires_cuda
def test_no_weight_only_kernel_serves_int8_configs():
    """Until a real int8-activation kernel exists, selection must raise so
    the scheme can fall back with a warning instead of silently running
    W4A16."""
    from vllm.model_executor.kernels.linear import choose_mp_linear_kernel

    with pytest.raises(ValueError):
        choose_mp_linear_kernel(_int8_cfg())
