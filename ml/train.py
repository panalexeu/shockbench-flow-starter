"""Behaviour cloning: train the student MLP on recorded (observation, teacher action) weeks, export an agent folder.

    uv run python -m ml.train --data=outputs/ml/collect/small_r0
    uv run python -m ml.train --data=outputs/ml/collect/small_r0 --hidden=1024 --epochs=100 --name=il_small
    uv run python -m ml.train --data=outputs/ml/collect/small_r0,outputs/ml/dagger/<run>/round_1   # several folders

Needs torch (``uv sync --extra rl``) for training only; the exported agent runs in numpy. Episodes are split into
train and validation by episode, never by week. The agent folder ``agents/<name>/`` gets agent.py (ml/student_agent.py),
features.py and student.npz (the layers' weights, and the features' mean, std and clip); a run's log goes to
``outputs/ml/train/<date_time>/``.
"""

import datetime
import json
import shutil
import time
from pathlib import Path

import fire
import numpy as np
import torch

from ml.features import Features


ML = Path(__file__).resolve().parent
CLIP = 10.0  # standardised features are clipped to [-CLIP, CLIP]


def load(data: str) -> tuple[Features, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Features, config and per-week arrays (X, Y, modes, episode id) of every episode in ``data`` (cached there)."""
    config = json.loads((Path(data) / "config.json").read_text())
    F = Features(config)
    cache = Path(data) / "features.npz"
    files = sorted(Path(data).glob("*_*_*.npz"))
    if cache.exists() and int(np.load(cache)["n_files"]) == len(files):
        c = np.load(cache)
        return F, config, c["X"], c["Y"], c["modes"], c["episode"]
    X, Y, modes, ep = [], [], [], []
    for i, f in enumerate(files):
        d = np.load(f)
        obs = {k.removeprefix("obs/"): d[k] for k in d.files if k.startswith("obs/")}
        T = d["act/flows"].shape[0]
        X.append(np.stack([F.vector({k: v[t] for k, v in obs.items()}) for t in range(T)]))
        Y.append(F.targets(d["act/flows"], d["act/override_qty"]))
        modes.append(d["act/release_mode"])
        ep.append(np.full(T, i))
    X, Y, modes, ep = (np.concatenate(a) for a in (X, Y, modes, ep))
    np.savez(cache, X=X, Y=Y, modes=modes, episode=ep, n_files=len(files))
    return F, config, X, Y, modes, ep


def load_many(data) -> tuple[Features, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``load`` of every folder in ``data`` (a comma-separated string or a list), episodes numbered across folders."""
    dirs = [d for d in (data.split(",") if isinstance(data, str) else data) if d]
    parts = [load(d) for d in dirs]
    F, config = parts[0][0], parts[0][1]
    X, Y, modes, ep, offset = [], [], [], [], 0
    for _F, _c, x, y, m, e in parts:
        X.append(x), Y.append(y), modes.append(m), ep.append(e + offset)
        offset += int(e.max()) + 1
    return F, config, *(np.concatenate(a) for a in (X, Y, modes, ep))


class MLP(torch.nn.Module):
    def __init__(self, n_in: int, n_y: int, n_pairs: int, hidden: int, layers: int) -> None:
        super().__init__()
        dims = [n_in] + [hidden] * layers
        self.body = torch.nn.ModuleList(torch.nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        self.out = torch.nn.Linear(hidden, n_y + 3 * n_pairs)
        self.n_y, self.n_pairs = n_y, n_pairs

    def forward(self, x):
        for lin in self.body:
            x = torch.relu(lin(x))
        z = self.out(x)
        return z[:, : self.n_y], z[:, self.n_y :].reshape(-1, self.n_pairs, 3)


def main(
    data: str = "outputs/ml/collect/small_r0", 
    name: str = "il_small", 
    hidden: int = 512, 
    layers: int = 2,
    epochs: int = 60, 
    lr: float = 1e-3, 
    batch: int = 256, 
    val_frac: float = 0.1, 
    mode_weight: float = 0.1,
    seed: int = 0
):
    run = Path(f"outputs/ml/train/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    F, config, X, Y, modes, ep = load_many(data)
    print(f"{len(np.unique(ep))} episodes, {len(X)} weeks, {X.shape[1]} features, {Y.shape[1]} targets "
          f"({time.perf_counter() - t0:.0f} s to load)")
    episodes = np.unique(ep)
    val_eps = rng.choice(episodes, size=max(1, int(len(episodes) * val_frac)), replace=False)
    is_val = np.isin(ep, val_eps)
    mean = X[~is_val].mean(axis=0)
    std = X[~is_val].std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    Z = np.clip((X - mean) / std, -CLIP, CLIP).astype(np.float32)
    tensors = [torch.from_numpy(a) for a in (Z, Y.astype(np.float32), modes.astype(np.int64))]
    tr = [t[torch.from_numpy(~is_val)] for t in tensors]
    va = [t[torch.from_numpy(is_val)] for t in tensors]
    model = MLP(Z.shape[1], Y.shape[1], modes.shape[1], hidden, layers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ce = torch.nn.CrossEntropyLoss()

    def losses(x, y, m):
        py, pm = model(x)
        return torch.mean((py - y) ** 2), ce(pm.reshape(-1, 3), m.reshape(-1))
             # ^ mse                      ^ cross entropy 
    best, log = None, []
    for e in range(epochs):
        model.train()
        perm = torch.randperm(len(tr[0]))
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            l_y, l_m = losses(tr[0][idx], tr[1][idx], tr[2][idx])
            opt.zero_grad()
            (l_y + mode_weight * l_m).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            ty, tm = losses(*tr)
            vy, vm = losses(*va)
            acc = (model(va[0])[1].argmax(-1) == va[2]).float().mean().item()
        row = {"epoch": e, "train_mse": ty.item(), "val_mse": vy.item(), "val_mode_ce": vm.item(), "val_mode_acc": acc}
        log.append(row)
        if best is None or vy.item() < best[0]:
            best = (vy.item(), {k: v.clone() for k, v in model.state_dict().items()}, e)
        if e % 10 == 0 or e == epochs - 1:
            print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
    model.load_state_dict(best[1])

    # export: numpy weights (x @ W + b) and the standardisation the agent applies first
    out = Path("agents") / name
    out.mkdir(parents=True, exist_ok=True)
    lins = list(model.body) + [model.out]
    Ws = [lin.weight.detach().numpy().T for lin in lins]
    bs = [lin.bias.detach().numpy() for lin in lins]
    np.savez(out / "student.npz", **{f"W{i}": w.astype(np.float32) for i, w in enumerate(Ws)},
             **{f"b{i}": b.astype(np.float32) for i, b in enumerate(bs)},
             mean=mean.astype(np.float32), std=std.astype(np.float32), clip=CLIP)
    # the exported numpy network must give the trained network's outputs
    x = np.clip((X[:256] - mean) / std, -CLIP, CLIP)
    for i, (w, b) in enumerate(zip(Ws, bs)):
        x = x @ w + b
        x = np.maximum(x, 0.0) if i < len(Ws) - 1 else x
    with torch.no_grad():
        py, pm = model(torch.from_numpy(Z[:256]))
    gap = np.abs(x - np.concatenate([py.numpy(), pm.reshape(len(py), -1).numpy()], axis=1)).max()
    print(f"export check: max |numpy - torch| = {gap:.2e}")
    shutil.copy(ML / "student_agent.py", out / "agent.py")
    shutil.copy(ML / "features.py", out / "features.py")
    meta = {"data": data, "name": name, "hidden": hidden, "layers": layers, "epochs": epochs, "lr": lr,
            "batch": batch, "seed": seed, "best_epoch": best[2], "best_val_mse": best[0],
            "weeks": int(len(X)), "val_episodes": [int(v) for v in val_eps], "log": log}
    (run / "train.json").write_text(json.dumps(meta, indent=1))
    print(f"best epoch {best[2]}, val mse {best[0]:.5f}; agent written to {out}; log {run / 'train.json'}")

if __name__ == "__main__":
    fire.Fire(main)
