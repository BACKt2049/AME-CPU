from __future__ import annotations

import torch

from ._harness import compile_tileops_kernel, get_kernel_class


def _reference(states: torch.Tensor, dA_chunk_cumsum: torch.Tensor, initial_states: torch.Tensor):
    out = []
    s = initial_states.float().clone()
    for ci in range(states.shape[1]):
        out.append(s.clone())
        s = torch.exp(dA_chunk_cumsum[:, :, ci]).unsqueeze(-1) * s + states[:, ci].float()
    return torch.stack(out, dim=1), s


def _run_case(vectorize: bool) -> None:
    batch, num_chunks, n_heads, d_state = 1, 3, 2, 8
    block_d = 8
    threads = 4 if vectorize else 8
    kernel_cls = get_kernel_class("mamba.ssd_state_passing", "SSDStatePassingFwdKernel")
    tileops_kernel = kernel_cls(
        batch,
        num_chunks,
        n_heads,
        d_state,
        has_initial_states=True,
        dtype=torch.float32,
        config={"block_d": block_d, "threads": threads, "vectorize": vectorize},
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    states = torch.linspace(-0.5, 0.5, batch * num_chunks * n_heads * d_state, dtype=torch.float32).reshape(
        batch, num_chunks, n_heads, d_state
    )
    dA_chunk_cumsum = -torch.linspace(
        0.01, 0.06, batch * n_heads * num_chunks, dtype=torch.float32
    ).reshape(batch, n_heads, num_chunks)
    initial_states = torch.linspace(
        0.1, 0.8, batch * n_heads * d_state, dtype=torch.float32
    ).reshape(batch, n_heads, d_state)

    actual_out, actual_final = kernel(states, dA_chunk_cumsum, initial_states)
    expected_out, expected_final = _reference(states, dA_chunk_cumsum, initial_states)
    torch.testing.assert_close(actual_out, expected_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_final, expected_final, rtol=1e-5, atol=1e-5)


def test_ssd_state_passing_float32_runtime_compare():
    _run_case(vectorize=False)


def test_ssd_state_passing_float32_vectorize_runtime_compare():
    _run_case(vectorize=True)
