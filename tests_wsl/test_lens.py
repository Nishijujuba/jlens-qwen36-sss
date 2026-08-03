from __future__ import annotations

import numpy as np
import torch
from torch import nn

from jlens_wsl.lens import TransportLens, readout_last_position
from tests_wsl.fakes import FakeTokenizer


def test_identity_transport_is_logit_lens():
    lens = TransportLens.identity()
    hidden = torch.tensor([[1.0, 2.0]])
    assert lens.mode == "logit"
    assert torch.equal(lens.transport(hidden, 0), hidden)


def test_npz_transport_and_direction(tmp_path):
    path = tmp_path / "lens.npz"
    matrix = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    np.savez(path, J_0=matrix, d_model=2, n_prompts=4)
    lens = TransportLens.load(path, expected_d_model=2)
    hidden = torch.tensor([[1.0, 2.0]])
    assert lens.mode == "jacobian"
    assert torch.allclose(lens.transport(hidden, 0), torch.tensor([[2.0, 6.0]]))
    row = torch.tensor([1.0, 1.0])
    assert torch.allclose(lens.concept_direction(row, 0), torch.tensor([2.0, 3.0]))


def test_batched_last_position_readout():
    tokenizer = FakeTokenizer()
    captured = {
        0: torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        1: torch.tensor([[[1.0, 1.0], [1.0, 0.0]]]),
    }
    head = nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]))
    readouts = readout_last_position(
        captured,
        layers=[0, 1],
        final_norm=nn.Identity(),
        lm_head=head,
        tokenizer=tokenizer,
        lens=TransportLens.identity(),
        top_k=2,
    )
    assert [item.layer for item in readouts] == [0, 1]
    assert readouts[0].top[0].token_id == 1
    assert readouts[1].top[0].token_id == 0
