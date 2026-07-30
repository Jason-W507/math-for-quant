from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class TextAdaptationResult:
    token_count: int
    full_parameters: int
    peft_trainable_parameters: int
    zero_shot_scores: np.ndarray
    few_shot_scores: np.ndarray
    peft_scores: np.ndarray


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
) -> TextAdaptationResult:
    labels = np.asarray(train_labels, dtype=np.float32)
    if labels.shape != (len(train_texts),) or not inference_texts:
        raise ValueError("text adaptation inputs are misaligned")
    vocabulary = sorted({token for text in train_texts for token in _tokens(text)})
    index = {token: position for position, token in enumerate(vocabulary)}
    width = 8
    torch.manual_seed(seed)
    embedding = nn.Embedding(len(vocabulary), width)
    for parameter in embedding.parameters():
        parameter.requires_grad_(False)

    def encode(texts: list[str]) -> torch.Tensor:
        rows = []
        for text in texts:
            ids = [index[token] for token in _tokens(text) if token in index]
            if not ids:
                rows.append(torch.zeros(width))
            else:
                rows.append(embedding(torch.tensor(ids)).mean(dim=0))
        return torch.stack(rows)

    positive = {"profit", "growth", "upgrade"}
    negative = {"loss", "warning", "decline"}
    zero_shot = np.asarray([
        sum(token in positive for token in _tokens(text))
        - sum(token in negative for token in _tokens(text))
        for text in inference_texts
    ], dtype=float)
    with torch.no_grad():
        train_encoded = encode(train_texts)
        inference_encoded = encode(inference_texts)
        positive_centroid = train_encoded[torch.from_numpy(labels == 1)].mean(dim=0)
        negative_centroid = train_encoded[torch.from_numpy(labels == 0)].mean(dim=0)
        few_shot = ((inference_encoded - negative_centroid) ** 2).sum(dim=1) - (
            (inference_encoded - positive_centroid) ** 2
        ).sum(dim=1)
    head = nn.Linear(width, 1)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    loss_function = nn.BCEWithLogitsLoss()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(head(train_encoded).squeeze(1), torch.from_numpy(labels))
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        peft = torch.sigmoid(head(inference_encoded).squeeze(1))
    full_parameters = sum(parameter.numel() for parameter in embedding.parameters()) + sum(
        parameter.numel() for parameter in head.parameters()
    )
    peft_parameters = sum(parameter.numel() for parameter in head.parameters())
    return TextAdaptationResult(
        token_count=len(vocabulary),
        full_parameters=full_parameters,
        peft_trainable_parameters=peft_parameters,
        zero_shot_scores=zero_shot,
        few_shot_scores=few_shot.numpy(),
        peft_scores=peft.numpy(),
    )
