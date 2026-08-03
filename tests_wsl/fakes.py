from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn


class FakeTokenizer:
    eos_token_id = 7
    pad_token_id = 0

    def __init__(self) -> None:
        self._chars = {" ": 1, "a": 2, "b": 3, "c": 4, "d": 5, "e": 6}
        self._inverse = {value: key for key, value in self._chars.items()}
        self._inverse[0] = ""
        self._inverse[7] = "<eos>"

    def encode(self, text, add_special_tokens=False):
        return [self._chars.get(char.lower(), 6) for char in text]

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        }

    def apply_chat_template(self, messages, **kwargs):
        text = " ".join(message["content"] for message in messages)
        return torch.tensor([self.encode(text)], dtype=torch.long)

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        output = ""
        for token_id in ids:
            token_id = int(token_id)
            if token_id == self.eos_token_id and skip_special_tokens:
                continue
            output += self._inverse.get(token_id, "?")
        return output


class FakeLayer(nn.Module):
    def __init__(self, width: int, scale: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale), requires_grad=False)
        self.proj = nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(width))

    def forward(self, hidden, **kwargs):
        return hidden + self.scale * torch.tanh(self.proj(hidden))


class FakeTextModel(nn.Module):
    def __init__(self, vocab: int = 8, width: int = 4, layers: int = 3) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, width)
        self.layers = nn.ModuleList([FakeLayer(width, 0.1 * (index + 1)) for index in range(layers)])
        self.norm = nn.LayerNorm(width)
        with torch.no_grad():
            values = torch.arange(vocab * width, dtype=torch.float32).reshape(vocab, width)
            self.embed_tokens.weight.copy_((values % 7 - 3) / 4)


class FakeBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = FakeTextModel()
        self.visual = nn.Linear(1, 1)


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeBase()
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.model.language_model.embed_tokens.weight
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(
                layer_types=["linear_attention", "linear_attention", "full_attention"]
            )
        )
        self.generation_config = SimpleNamespace(eos_token_id=7)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        use_cache=False,
        logits_to_keep=1,
        return_dict=True,
        **kwargs,
    ):
        hidden = self.model.language_model.embed_tokens(input_ids)
        for layer in self.model.language_model.layers:
            hidden = layer(hidden)
        hidden = self.model.language_model.norm(hidden)
        if logits_to_keep:
            hidden_for_logits = hidden[:, -logits_to_keep:, :]
        else:
            hidden_for_logits = hidden
        logits = self.lm_head(hidden_for_logits)
        return SimpleNamespace(logits=logits)
