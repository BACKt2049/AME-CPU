from __future__ import annotations

import torch
import torch.nn.functional as F

from ._harness import compile_tileops_kernel, get_kernel_class


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float):
    x_f = x.float()
    rrms = (x_f * x_f).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return x_f * rrms * weight.float(), rrms.squeeze(-1)


def _forward_reference(
    hidden: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    rms_w_h: torch.Tensor,
    rms_w_v: torch.Tensor,
    conv_w: torch.Tensor,
    eps: float,
):
    _, _, hidden_dim = hidden.shape
    hidden_norm, rrms_h = _rmsnorm(hidden, rms_w_h, eps)
    key_norm, rrms_k = _rmsnorm(key, rms_w_h, eps)
    alpha = torch.sigmoid((hidden_norm * key_norm).sum(dim=-1, keepdim=True) / (hidden_dim**0.5))
    vhat = alpha * value.float()
    vhat_norm, rrms_v = _rmsnorm(vhat.to(hidden.dtype), rms_w_v, eps)
    conv_in = F.pad(vhat_norm.float().permute(0, 2, 1), (conv_w.shape[0] - 1, 0))
    conv_out = F.conv1d(conv_in, conv_w.float().T.unsqueeze(1), groups=hidden_dim).permute(0, 2, 1)
    output = F.silu(conv_out) + vhat.float()
    return (
        output.to(hidden.dtype),
        vhat.to(hidden.dtype),
        alpha.squeeze(-1).float(),
        rrms_h.float(),
        rrms_k.float(),
        rrms_v.float(),
    )


def test_engram_gate_conv_bwd_float32_runtime_compare():
    batch, seq_len, hidden_dim = 1, 4, 256
    eps = 1e-6
    kernel_cls = get_kernel_class("engram.engram_bwd", "EngramGateConvBwdKernel")
    tileops_kernel = kernel_cls(
        batch,
        seq_len,
        hidden_dim,
        eps,
        torch.float32,
        config={"threads": 128},
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    hidden = torch.linspace(-0.5, 0.5, batch * seq_len * hidden_dim, dtype=torch.float32).reshape(
        batch, seq_len, hidden_dim
    )
    key = torch.linspace(-0.4, 0.4, batch * seq_len * hidden_dim, dtype=torch.float32).reshape(
        batch, seq_len, hidden_dim
    )
    value = torch.linspace(-0.3, 0.3, batch * seq_len * hidden_dim, dtype=torch.float32).reshape(
        batch, seq_len, hidden_dim
    )
    rms_w_h = torch.linspace(0.8, 1.2, hidden_dim, dtype=torch.float32)
    rms_w_v = torch.linspace(0.9, 1.1, hidden_dim, dtype=torch.float32)
    conv_w = torch.linspace(-0.2, 0.2, 4 * hidden_dim, dtype=torch.float32).reshape(4, hidden_dim)

    hidden_ref = hidden.clone().requires_grad_()
    key_ref = key.clone().requires_grad_()
    value_ref = value.clone().requires_grad_()
    rms_w_h_ref = rms_w_h.clone().requires_grad_()
    rms_w_v_ref = rms_w_v.clone().requires_grad_()
    conv_w_ref = conv_w.clone().requires_grad_()
    expected_out, vhat, alpha, rrms_h, rrms_k, rrms_v = _forward_reference(
        hidden_ref,
        key_ref,
        value_ref,
        rms_w_h_ref,
        rms_w_v_ref,
        conv_w_ref,
        eps,
    )
    dout = torch.linspace(-0.25, 0.25, batch * seq_len * hidden_dim, dtype=torch.float32).reshape(
        batch, seq_len, hidden_dim
    )
    expected_out.backward(dout)

    actual = kernel(
        dout,
        hidden,
        key,
        value,
        rms_w_h,
        rms_w_v,
        conv_w,
        vhat.detach(),
        alpha.detach(),
        rrms_h.detach(),
        rrms_k.detach(),
        rrms_v.detach(),
    )
    expected = [
        hidden_ref.grad,
        key_ref.grad,
        value_ref.grad,
        rms_w_h_ref.grad,
        rms_w_v_ref.grad,
        conv_w_ref.grad,
    ]
    for actual_tensor, expected_tensor in zip(actual[:6], expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=2e-4, atol=2e-4)
