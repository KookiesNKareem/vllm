# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dp4a W4A8 linear kernel for Ampere-class GPUs (sm_80..sm_89)."""

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

_GROUP_SIZE = 128
_MAX_GEMV_BATCH = 8


def _pack_w4(weight: torch.Tensor) -> torch.Tensor:
    """Pack s4 values (int8 [N, K]) into int32 [N, K/8], byte-nibble order.

    Word byte b holds w[8u + b] in its low nibble and w[8u + 4 + b] in its
    high nibble, matching the dp4a kernel's high-nibble unpack.
    """
    nib = (weight.to(torch.int32) & 0xF)
    packed = torch.zeros(
        weight.shape[0], weight.shape[1] // 8, dtype=torch.int32, device=weight.device
    )
    for b in range(4):
        packed |= nib[:, b::8] << (8 * b)
        packed |= nib[:, 4 + b::8] << (8 * b + 4)
    return packed


class Dp4aW4A8LinearKernel(MPLinearKernel):
    """Decode-optimized W4A8 (int4 weights, per-token dynamic int8
    activations) via dp4a. Batches above _MAX_GEMV_BATCH dequantize and use
    a dense GEMM."""

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "dp4a W4A8 requires CUDA"
        capability = current_platform.get_device_capability()
        if capability is None or not (80 <= capability.to_int() < 90):
            return False, "dp4a W4A8 targets Ampere/Ada (sm_80..sm_89)"
        if c.act_type != torch.int8:
            return False, "dp4a W4A8 requires int8 activations"
        if c.weight_type != scalar_types.int4:
            return False, "dp4a W4A8 requires signed int4 weights"
        if c.group_size != _GROUP_SIZE:
            return False, f"dp4a W4A8 requires group_size={_GROUP_SIZE}"
        if c.zero_points:
            return False, "dp4a W4A8 requires symmetric weights"
        if c.has_g_idx:
            return False, "dp4a W4A8 does not support act-order (g_idx)"
        if c.partition_weight_shape[0] % _GROUP_SIZE != 0:
            return False, "K must be a multiple of the group size"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._transform_param(layer, self.w_q_name, lambda w: _pack_w4(w.data))
        # the kernel's dp4a operands are 16 * s4; fold the /16 into the scales
        self._transform_param(
            layer,
            self.w_s_name,
            lambda s: (s.data.to(torch.float32) / 16.0).to(s.dtype),
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q = getattr(layer, self.w_q_name)
        w_s = getattr(layer, self.w_s_name)
        x2d = x.reshape(-1, x.shape[-1])
        bs = x2d.shape[0]
        out_shape = x.shape[:-1] + (w_q.shape[0],)

        if bs <= _MAX_GEMV_BATCH:
            x_q, x_s, _ = ops.scaled_int8_quant(x2d, symmetric=True)
            padded = 1 << (bs - 1).bit_length() if bs > 1 else 1
            if padded != bs:
                x_q = torch.nn.functional.pad(x_q, (0, 0, 0, padded - bs))
                x_s = torch.nn.functional.pad(x_s.view(-1), (0, padded - bs))
            out = torch.empty(
                padded, w_q.shape[0], dtype=x.dtype, device=x.device
            )
            ops.w4a8_dp4a_gemm(out, x_q, x_s.view(-1).float(), w_q, w_s)
            out = out[:bs]
        else:
            w_fp = torch.empty(
                w_q.shape[0], w_q.shape[1] * 8, dtype=x.dtype, device=x.device
            )
            ops.w4a8_dp4a_dequant(w_fp, w_q, w_s)
            out = torch.mm(x2d, w_fp.t())

        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)
