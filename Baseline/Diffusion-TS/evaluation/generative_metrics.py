"""Unified metrics shared with the Diffusion-PETS comparison notebooks."""

import random

import numpy as np
import scipy.linalg
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from Models.ts2vec.ts2vec import TS2Vec
from Utils.context_fid import calculate_fid
from Utils.cross_correlation import CrossCorrelLoss


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _balanced(real, fake, max_samples, seed):
    rng = np.random.default_rng(seed)
    count = min(len(real), len(fake), max_samples)
    real_idx = rng.choice(len(real), count, replace=False)
    fake_idx = rng.choice(len(fake), count, replace=False)
    return (
        np.asarray(real[real_idx], dtype=np.float32),
        np.asarray(fake[fake_idx], dtype=np.float32),
    )


class _GRUDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden = max(input_dim // 2, 4)
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.head(hidden[-1]).squeeze(-1)


class _GRUPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden = max(input_dim // 2, 4)
        self.gru = nn.GRU(input_dim, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        output, _ = self.gru(x)
        return torch.sigmoid(self.head(output))


def correlational_score(real, fake, max_samples=2048, seed=0):
    real, fake = _balanced(real, fake, max_samples, seed)
    loss = CrossCorrelLoss(torch.from_numpy(real), name="CrossCorrelLoss")
    return float(loss.compute(torch.from_numpy(fake)).item())


def context_fid(real, fake, device):
    """Run the original Diffusion-TS Context-FID with a selectable device."""
    model = TS2Vec(
        input_dims=real.shape[-1],
        device=device,
        batch_size=8,
        lr=0.001,
        output_dims=320,
        max_train_length=3000,
    )
    model.fit(real, verbose=False)
    real_representation = model.encode(real, encoding_window="full_series")
    fake_representation = model.encode(fake, encoding_window="full_series")
    indices = np.random.permutation(real.shape[0])
    return float(
        calculate_fid(
            real_representation[indices],
            fake_representation[indices],
        )
    )


def discriminative_score(
    real,
    fake,
    device,
    seed=0,
    iterations=2000,
    batch_size=128,
    max_samples=4096,
):
    _seed(seed)
    real, fake = _balanced(real, fake, max_samples, seed)
    data = np.concatenate([real, fake])
    labels = np.concatenate([np.ones(len(real)), np.zeros(len(fake))]).astype(np.float32)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data))
    split = int(0.8 * len(order))
    train_idx, test_idx = order[:split], order[split:]
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data[train_idx]), torch.from_numpy(labels[train_idx])),
        batch_size=min(batch_size, len(train_idx)),
        shuffle=True,
    )
    model = _GRUDiscriminator(data.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    iterator = iter(loader)
    for _ in range(iterations):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(data[test_idx]).to(device)).cpu()
    accuracy = float(((logits > 0).numpy() == labels[test_idx]).mean())
    return abs(accuracy - 0.5)


def predictive_score(
    real,
    fake,
    device,
    seed=0,
    iterations=5000,
    batch_size=128,
    max_samples=4096,
):
    _seed(seed)
    real, fake = _balanced(real, fake, max_samples, seed)
    input_dim = max(fake.shape[-1] - 1, 1)
    fake_x = torch.from_numpy(fake[:, :-1, :input_dim])
    fake_y = torch.from_numpy(fake[:, 1:, -1:])
    loader = DataLoader(
        TensorDataset(fake_x, fake_y),
        batch_size=min(batch_size, len(fake_x)),
        shuffle=True,
    )
    model = _GRUPredictor(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()
    model.train()
    iterator = iter(loader)
    for _ in range(iterations):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(
            torch.from_numpy(real[:, :-1, :input_dim]).to(device)
        ).cpu().numpy()
    return float(np.mean(np.abs(prediction - real[:, 1:, -1:])))


def evaluate_generation(real, fake, gpu, seed, max_samples=4096):
    real, fake = _balanced(real, fake, max_samples, seed)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    return {
        "context_fid": context_fid(
            real,
            fake,
            device=gpu if torch.cuda.is_available() else "cpu",
        ),
        "correlational_score": correlational_score(
            real,
            fake,
            max_samples=max_samples,
            seed=seed,
        ),
        "discriminative_score": discriminative_score(
            real,
            fake,
            device=device,
            seed=seed,
        ),
        "predictive_score": predictive_score(
            real,
            fake,
            device=device,
            seed=seed,
        ),
    }
