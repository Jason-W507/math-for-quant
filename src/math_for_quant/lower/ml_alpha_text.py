from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import copy

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class TextAdaptationResult:
    token_count: int
    encoder_id: str
    encoder_version: str
    encoder_license: str
    full_trainable_parameters: int
    lora_trainable_parameters: int
    zero_shot_scores: np.ndarray
    few_shot_scores: np.ndarray
    full_finetune_scores: np.ndarray
    lora_scores: np.ndarray


class TinyTextEncoder(nn.Module):
    def __init__(self, vocabulary_size: int, width: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, width)
        self.projection = nn.Linear(width, width)

    def forward(self, bags: torch.Tensor) -> torch.Tensor:
        normalized = bags / bags.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = normalized @ self.embedding.weight
        return torch.tanh(self.projection(pooled))


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int) -> None:
        super().__init__()
        self.register_buffer("base_weight", base.weight.detach().clone())
        self.register_buffer("base_bias", base.bias.detach().clone())
        self.adapter_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.adapter_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.normal_(self.adapter_a, mean=0.0, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        adapted_weight = self.base_weight + self.adapter_b @ self.adapter_a
        return nn.functional.linear(values, adapted_weight, self.base_bias)


class LoRATextClassifier(nn.Module):
    def __init__(self, encoder: TinyTextEncoder, *, rank: int) -> None:
        super().__init__()
        self.register_buffer("embedding_weight", encoder.embedding.weight.detach().clone())
        self.projection = LoRALinear(encoder.projection, rank=rank)
        self.head = nn.Linear(encoder.projection.out_features, 1)

    def forward(self, bags: torch.Tensor) -> torch.Tensor:
        normalized = bags / bags.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = normalized @ self.embedding_weight
        return self.head(torch.tanh(self.projection(pooled))).squeeze(1)


def audit_text_timestamps(
    *, publication_dates: list[str], revision_dates: list[str], decision_date: str
) -> None:
    if len(publication_dates) != len(revision_dates) or not publication_dates:
        raise ValueError("text timestamp rows are misaligned")
    decision = date.fromisoformat(decision_date)
    for publication, revision in zip(publication_dates, revision_dates, strict=True):
        if date.fromisoformat(publication) > decision:
            raise ValueError("publication leakage crosses the decision date")
        if date.fromisoformat(revision) > decision:
            raise ValueError("revision leakage crosses the decision date")


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in text.split() if token]


def compare_text_adaptation(
    *,
    train_texts: list[str],
    train_labels: np.ndarray,
    inference_texts: list[str],
    seed: int,
    encoder_id: str = "mfq-tiny-text-encoder",
    encoder_version: str = "1.0.0",
    encoder_license: str = "CC0-1.0",
) -> TextAdaptationResult:
    labels = np.asarray(train_labels, dtype=np.float32)
    if labels.shape != (len(train_texts),) or not inference_texts:
        raise ValueError("text adaptation inputs are misaligned")
    vocabulary = sorted({token for text in train_texts for token in _tokens(text)})
    index = {token: position for position, token in enumerate(vocabulary)}
    width = 8
    torch.manual_seed(seed)

    def bag(texts: list[str]) -> torch.Tensor:
        rows = torch.zeros((len(texts), len(vocabulary)))
        for row, text in enumerate(texts):
            for token in _tokens(text):
                if token in index:
                    rows[row, index[token]] += 1.0
        return rows

    train_bags = bag(train_texts)
    inference_bags = bag(inference_texts)
    encoder = TinyTextEncoder(len(vocabulary), width)
    decoder = nn.Linear(width, len(vocabulary))
    pretraining_optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=0.03
    )
    reconstruction_loss = nn.BCEWithLogitsLoss()
    reconstruction_target = (train_bags > 0.0).float()
    for _ in range(120):
        pretraining_optimizer.zero_grad(set_to_none=True)
        loss = reconstruction_loss(decoder(encoder(train_bags)), reconstruction_target)
        loss.backward()
        pretraining_optimizer.step()

    positive = {"profit", "growth", "upgrade"}
    negative = {"loss", "warning", "decline"}
    zero_shot = np.asarray([
        sum(token in positive for token in _tokens(text))
        - sum(token in negative for token in _tokens(text))
        for text in inference_texts
    ], dtype=float)
    with torch.no_grad():
        train_encoded = encoder(train_bags)
        inference_encoded = encoder(inference_bags)
        positive_centroid = train_encoded[torch.from_numpy(labels == 1)].mean(dim=0)
        negative_centroid = train_encoded[torch.from_numpy(labels == 0)].mean(dim=0)
        few_shot = ((inference_encoded - negative_centroid) ** 2).sum(dim=1) - (
            (inference_encoded - positive_centroid) ** 2
        ).sum(dim=1)
    full_encoder = copy.deepcopy(encoder)
    full_head = nn.Linear(width, 1)
    full_model = nn.Sequential(full_encoder, full_head)
    optimizer = torch.optim.SGD(full_model.parameters(), lr=0.1)
    loss_function = nn.BCEWithLogitsLoss()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(full_model(train_bags).squeeze(1), torch.from_numpy(labels))
        loss.backward()
        optimizer.step()
    lora_model = LoRATextClassifier(copy.deepcopy(encoder), rank=2)
    lora_optimizer = torch.optim.SGD(
        (parameter for parameter in lora_model.parameters() if parameter.requires_grad),
        lr=0.1,
    )
    for _ in range(80):
        lora_optimizer.zero_grad(set_to_none=True)
        loss = loss_function(lora_model(train_bags), torch.from_numpy(labels))
        loss.backward()
        lora_optimizer.step()
    with torch.no_grad():
        full_finetune = torch.sigmoid(full_model(inference_bags).squeeze(1))
        lora = torch.sigmoid(lora_model(inference_bags))
    full_trainable = sum(
        parameter.numel() for parameter in full_model.parameters() if parameter.requires_grad
    )
    lora_trainable = sum(
        parameter.numel() for parameter in lora_model.parameters() if parameter.requires_grad
    )
    return TextAdaptationResult(
        token_count=len(vocabulary),
        encoder_id=encoder_id,
        encoder_version=encoder_version,
        encoder_license=encoder_license,
        full_trainable_parameters=full_trainable,
        lora_trainable_parameters=lora_trainable,
        zero_shot_scores=zero_shot,
        few_shot_scores=few_shot.numpy(),
        full_finetune_scores=full_finetune.numpy(),
        lora_scores=lora.numpy(),
    )
