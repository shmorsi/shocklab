"""Mixture density network: p(asset returns | factor moves, regime).

    p(y | x) = sum_k pi_k(x) * prod_j N(y_j; mu_kj(x), sigma_kj(x)^2)

K=5 components, each a diagonal Gaussian over all assets; cross-asset correlation
comes from the mixture (components move assets together). Trained by minimising
the negative log-likelihood, with missing assets (short histories) masked out —
for a diagonal Gaussian, dropping a dimension is exactly marginalising it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from shocklab import config as C
from shocklab.data import Panel

K = 5
HIDDEN = 128
LOG_SIG_MIN, LOG_SIG_MAX = -5.0, 3.0  # in standardized units; keeps the NLL finite
LOG_2PI = math.log(2 * math.pi)


class MDNNet(nn.Module):
    def __init__(self, n_in: int, n_out: int, k: int = K) -> None:
        super().__init__()
        self.k, self.d = k, n_out
        self.body = nn.Sequential(
            nn.Linear(n_in, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
        )
        self.head = nn.Linear(HIDDEN, k + 2 * k * n_out)  # logits | means | log sigmas

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.head(self.body(x))
        logit = h[:, : self.k]
        mu = h[:, self.k : self.k + self.k * self.d].view(-1, self.k, self.d)
        log_sig = h[:, self.k + self.k * self.d :].view(-1, self.k, self.d)
        log_pi = torch.log_softmax(logit, dim=1)  # mixture weights sum to 1
        return log_pi, mu, log_sig.clamp(LOG_SIG_MIN, LOG_SIG_MAX)


def mdn_nll(log_pi, mu, log_sig, y, mask) -> torch.Tensor:
    """Mean over samples of -log sum_k pi_k prod_j N(y_j | mu_kj, sigma_kj), observed j only.

    Per dimension: log N = -0.5 * ((y - mu)/sigma)^2 - log sigma - 0.5 log(2 pi).
    logsumexp over k keeps the sum of tiny probabilities numerically stable.
    """
    y, mask = y.unsqueeze(1), mask.unsqueeze(1)  # broadcast over components
    z = (y - mu) / log_sig.exp()
    log_n = (-0.5 * z**2 - log_sig - 0.5 * LOG_2PI) * mask
    return -torch.logsumexp(log_pi + log_n.sum(dim=2), dim=1).mean()


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    def fwd(self, a: np.ndarray) -> np.ndarray:
        return (a - self.mean) / self.std


def features(panel: Panel) -> pd.DataFrame:
    """5 factor moves over the window + regime known at its start."""
    return pd.concat([panel.X[C.FACTORS], panel.R[["vix_level", "avg_corr"]]], axis=1)


@dataclass
class MDNModel:
    net: MDNNet
    assets: list[str]
    xs: Scaler  # input scaler
    ys: Scaler  # target scaler (per-asset mean/std from training data)
    train_end: str
    best_val_nll: float = float("nan")
    epochs: int = 0

    def _params(self, shock: np.ndarray, regime: np.ndarray):
        x = self.xs.fwd(np.concatenate([shock, regime]))[None, :]
        with torch.no_grad():
            log_pi, mu, log_sig = self.net(torch.as_tensor(x, dtype=torch.float32))
        return log_pi[0].numpy(), mu[0].numpy(), log_sig[0].numpy()

    def mixture(self, shock: np.ndarray, regime: np.ndarray):
        """Mixture weights, means and sds in raw 10-day log-return units."""
        log_pi, mu, log_sig = self._params(shock, regime)
        return np.exp(log_pi), mu * self.ys.std + self.ys.mean, np.exp(log_sig) * self.ys.std

    def mean(self, shock: np.ndarray, regime: np.ndarray) -> np.ndarray:
        pi, mu, _ = self.mixture(shock, regime)
        return pi @ mu

    def simulate(self, shock: np.ndarray, regime: np.ndarray, n: int = 10_000, seed: int = C.SEED) -> np.ndarray:
        """Pick a component per scenario by pi, then draw each asset from its Gaussian."""
        rng = np.random.default_rng(seed)
        pi, mu, sd = self.mixture(shock, regime)
        comp = rng.choice(len(pi), size=n, p=pi / pi.sum())
        return mu[comp] + sd[comp] * rng.standard_normal((n, len(self.assets)))

    def logpdf(self, y: np.ndarray, shock: np.ndarray, regime: np.ndarray) -> float:
        """Joint log density in raw units = standardized log density - sum log(std_j)."""
        log_pi, mu, log_sig = self._params(shock, regime)
        z = (self.ys.fwd(y) - mu) / np.exp(log_sig)
        log_n = (-0.5 * z**2 - log_sig - 0.5 * LOG_2PI).sum(axis=1)
        m = log_n.max() + log_pi.max()
        return float(m + np.log(np.exp(log_pi + log_n - m).sum()) - np.log(self.ys.std).sum())

    def save(self, path: Path) -> None:
        torch.save({"state": self.net.state_dict(), "assets": self.assets, "xs": self.xs.__dict__,
                    "ys": self.ys.__dict__, "train_end": self.train_end,
                    "best_val_nll": self.best_val_nll, "epochs": self.epochs}, path)

    @staticmethod
    def load(path: Path) -> "MDNModel":
        d = torch.load(path, weights_only=False)
        net = MDNNet(len(d["xs"]["mean"]), len(d["assets"]))
        net.load_state_dict(d["state"])
        net.eval()
        return MDNModel(net, d["assets"], Scaler(**d["xs"]), Scaler(**d["ys"]), d["train_end"],
                        d["best_val_nll"], d["epochs"])


def train_mdn(panel: Panel, train_end: str, val_end: str, seed: int = C.SEED, refit: bool = False,
              max_epochs: int = 400, patience: int = 25, lr: float = 1e-3, batch: int = 256) -> MDNModel:
    """Train on windows ending <= train_end; early-stop on windows that start after
    train_end and end <= val_end. Nothing after val_end is touched.

    refit=True: after early stopping picks the epoch count E, retrain from scratch on
    train+val for exactly E epochs, so the deployed model also learns from the most
    recent data (still nothing after val_end)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    feats = features(panel)
    tr = panel.known_before(train_end)
    va = (panel.Y.index > pd.Timestamp(train_end)) & panel.known_before(val_end)

    Xtr, Ytr = feats.to_numpy()[tr], panel.Y.to_numpy()[tr]
    xs = Scaler(Xtr.mean(0), Xtr.std(0))
    ys = Scaler(np.nanmean(Ytr, 0), np.nanstd(Ytr, 0))

    def tensors(rows):
        x = xs.fwd(feats.to_numpy()[rows])
        y = ys.fwd(panel.Y.to_numpy()[rows])
        mask = ~np.isnan(y)
        return (torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(np.nan_to_num(y), dtype=torch.float32),
                torch.as_tensor(mask, dtype=torch.float32))

    xt, yt, mt = tensors(tr)
    xv, yv, mv = tensors(va)
    net = MDNNet(xt.shape[1], yt.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)

    best, best_state, bad, epoch = float("inf"), None, 0, 0
    for epoch in range(1, max_epochs + 1):
        net.train()
        # Shuffling *within* the training set is fine: the time split is already done.
        for idx in torch.randperm(len(xt), generator=gen).split(batch):
            opt.zero_grad()
            loss = mdn_nll(*net(xt[idx]), yt[idx], mt[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = mdn_nll(*net(xv), yv, mv).item()
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    best_epoch = epoch - bad
    if refit:
        return _fit_fixed_epochs(panel, val_end, best_epoch, seed, lr, batch, best)
    net.load_state_dict(best_state)
    net.eval()
    return MDNModel(net, list(panel.Y.columns), xs, ys, str(train_end), best, best_epoch)


def _fit_fixed_epochs(panel: Panel, end: str, epochs: int, seed: int, lr: float, batch: int,
                      val_nll: float) -> MDNModel:
    """Retrain on every window ending <= end for a fixed number of epochs (no early stop)."""
    torch.manual_seed(seed)
    feats = features(panel)
    rows = panel.known_before(end)
    X, Y = feats.to_numpy()[rows], panel.Y.to_numpy()[rows]
    xs, ys = Scaler(X.mean(0), X.std(0)), Scaler(np.nanmean(Y, 0), np.nanstd(Y, 0))
    y = ys.fwd(Y)
    x_t = torch.as_tensor(xs.fwd(X), dtype=torch.float32)
    m_t = torch.as_tensor(~np.isnan(y), dtype=torch.float32)
    y_t = torch.as_tensor(np.nan_to_num(y), dtype=torch.float32)
    net = MDNNet(x_t.shape[1], y_t.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(x_t), generator=gen).split(batch):
            opt.zero_grad()
            mdn_nll(*net(x_t[idx]), y_t[idx], m_t[idx]).backward()
            opt.step()
    net.eval()
    return MDNModel(net, list(panel.Y.columns), xs, ys, str(end), val_nll, epochs)


def walk_forward_splits(year: int) -> tuple[str, str]:
    """Model used during `year`: train on windows ending by Dec 31 of year-3, validate
    on the following two years. year=2020 gives exactly train<=2017, val 2018-2019."""
    return f"{year - 3}-12-31", f"{year - 1}-12-31"


def model_path(year: int) -> Path:
    return C.ARTIFACTS_DIR / f"mdn_{year}.pt"


def load_for_date(date: pd.Timestamp, cache: dict[int, MDNModel] | None = None) -> MDNModel:
    """Walk-forward model for a date = the one trained before Jan 1 of that year."""
    cache = {} if cache is None else cache
    if date.year not in cache:
        cache[date.year] = MDNModel.load(model_path(date.year))
    return cache[date.year]
