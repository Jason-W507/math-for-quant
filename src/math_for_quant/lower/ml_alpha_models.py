from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class TorchTrainingConfig:
    seed: int
    epochs: int
    learning_rate: float
    device: str = "cpu"


@dataclass(frozen=True)
class TrainingArtifact:
    predictions: np.ndarray
    loss: float
    checkpoint_sha256: str
    checkpoint: bytes
    device: str
    batch_count: int


def _tiny_mlp(input_size: int) -> nn.Module:
    return nn.Sequential(nn.Linear(input_size, 8), nn.Tanh(), nn.Linear(8, 1))


def _serialize_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TorchTrainingConfig,
    input_size: int,
) -> bytes:
    stream = io.BytesIO()
    torch.save(
        {
            "format": "mfq-tiny-mlp-v1",
            "config": config.__dict__,
            "input_size": input_size,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        stream,
        _use_new_zipfile_serialization=False,
    )
    return stream.getvalue()


def _checkpoint_fingerprint(model: nn.Module, config: TorchTrainingConfig) -> str:
    digest = hashlib.sha256(
        json.dumps(config.__dict__, sort_keys=True).encode("utf-8")
    )
    for name, parameter in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def restore_tiny_mlp_predictions(
    checkpoint: bytes, features: np.ndarray, *, device: str = "cpu"
) -> np.ndarray:
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("requested CUDA device is unavailable")
    bundle = torch.load(io.BytesIO(checkpoint), map_location=requested_device, weights_only=False)
    if bundle.get("format") != "mfq-tiny-mlp-v1":
        raise ValueError("unsupported checkpoint format")
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != int(bundle["input_size"]):
        raise ValueError("checkpoint input shape is incompatible")
    model = _tiny_mlp(int(bundle["input_size"])).to(requested_device)
    model.load_state_dict(bundle["model_state"])
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(x).to(requested_device)).squeeze(1)
    return prediction.cpu().numpy().copy()


def train_tiny_mlp(
    features: np.ndarray,
    target: np.ndarray,
    *,
    config: TorchTrainingConfig,
) -> TrainingArtifact:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(target, dtype=np.float32)
    if x.ndim != 2 or y.shape != (x.shape[0],):
        raise ValueError("MLP features and target are misaligned")
    if config.epochs < 1 or config.learning_rate <= 0.0:
        raise ValueError("training configuration is invalid")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("requested CUDA device is unavailable")
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y[:, None]))
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    model = _tiny_mlp(x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.MSELoss()
    batch_count = 0
    model.train()
    for _ in range(config.epochs):
        for batch_x, batch_y in loader:
            batch_count = len(loader)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(x).to(device)).squeeze(1)
        final_loss = loss_function(prediction, torch.from_numpy(y).to(device))
    checkpoint = _serialize_checkpoint(model, optimizer, config, x.shape[1])
    return TrainingArtifact(
        predictions=prediction.cpu().numpy().copy(),
        loss=float(final_loss),
        checkpoint_sha256=_checkpoint_fingerprint(model, config),
        checkpoint=checkpoint,
        device=str(device),
        batch_count=batch_count,
    )


def _positional(values: torch.Tensor) -> torch.Tensor:
    steps = torch.arange(values.shape[1], dtype=values.dtype)[None, :, None]
    return values + 0.05 * steps


def sequence_order_sensitivity(sequence: np.ndarray, *, seed: int) -> dict[str, float]:
    values = torch.as_tensor(np.asarray(sequence, dtype=np.float32))
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[2] != 1:
        raise ValueError("sequence evidence expects shape (1, time, 1)")
    reversed_values = torch.flip(values, dims=[1])
    torch.manual_seed(seed)

    def gap(left: torch.Tensor, right: torch.Tensor) -> float:
        return float(torch.max(torch.abs(left - right)))

    convolution = nn.Conv1d(1, 1, kernel_size=2, bias=False)
    with torch.no_grad():
        convolution.weight[:] = torch.tensor([[[-1.0, 1.0]]])
        conv_left = convolution(values.transpose(1, 2))[:, :, -1]
        conv_right = convolution(reversed_values.transpose(1, 2))[:, :, -1]

    rnn = nn.RNN(input_size=1, hidden_size=3, batch_first=True)
    attention = nn.MultiheadAttention(embed_dim=4, num_heads=2, batch_first=True)
    projection = nn.Linear(1, 4, bias=False)
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=4, nhead=2, dim_feedforward=8, dropout=0.0, batch_first=True
    )
    transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
    causal_mask = torch.triu(
        torch.full((values.shape[1], values.shape[1]), float("-inf")), diagonal=1
    )
    with torch.no_grad():
        rnn_left = rnn(values)[0][:, -1]
        rnn_right = rnn(reversed_values)[0][:, -1]
        embedded_left = _positional(projection(values))
        embedded_right = _positional(projection(reversed_values))
        attention_left = attention(
            embedded_left, embedded_left, embedded_left, attn_mask=causal_mask
        )[0][:, -1]
        attention_right = attention(
            embedded_right, embedded_right, embedded_right, attn_mask=causal_mask
        )[0][:, -1]
        transformer_left = transformer(embedded_left, mask=causal_mask)[:, -1]
        transformer_right = transformer(embedded_right, mask=causal_mask)[:, -1]
    return {
        "mean_pool": gap(values.mean(dim=1), reversed_values.mean(dim=1)),
        "causal_conv": gap(conv_left, conv_right),
        "rnn": gap(rnn_left, rnn_right),
        "attention": gap(attention_left, attention_right),
        "transformer": gap(transformer_left, transformer_right),
    }
