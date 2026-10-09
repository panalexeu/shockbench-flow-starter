"""The student: an MLP trained to act like the perfect-foresight teacher (ml/PLAN.md). Copied as agent.py.

Inference is plain numpy. student.npz beside this file holds the layers' weights (W0, b0, W1, b1, ...) and the
features' standardisation (mean, std, clip), loaded at import (the CPU budget does not meter the import). It fits only
the network it was trained on.
"""

from pathlib import Path

import numpy as np

from features import Features


HERE = Path(__file__).resolve().parent
W = np.load(HERE / "student.npz", allow_pickle=False)
LAYERS = [(W[f"W{i}"], W[f"b{i}"]) for i in range(sum(k.startswith("W") for k in W.files))]
MEAN, STD, CLIP = W["mean"], W["std"], float(W["clip"])


class Agent:
    def __init__(self, config):
        self.features = Features(config)
        self.n_y = self.features.n_slots + self.features.n_ov

    def act(self, observation):
        x = np.clip((self.features.vector(observation) - MEAN) / STD, -CLIP, CLIP)
        for i, (w, b) in enumerate(LAYERS):
            x = x @ w + b
            if i < len(LAYERS) - 1:
                x = np.maximum(x, 0.0)
        return self.features.action(x[: self.n_y], x[self.n_y :], observation)
