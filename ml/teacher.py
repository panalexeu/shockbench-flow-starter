"""The teacher: ``mpc_det`` re-solved every week from the current state on the episode's true future (training only).

It reads the true weekly marks of the episode, which no submission can see, and plans over every remaining week with
the simulator's planning rules on. It answers "what would you do here?" from any state, so it can label the states a
student reaches (DAgger). Measured on Small dev: RSS ~0.80.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shockbench_flow import marks as M
from shockbench_flow.evaluation.cache import fq_quantiles
from shockbench_flow.hosting.tasks import scenario, task_generator
from shockbench_flow.marks import compute_marks
from shockbench_flow.policies import lp_common as L
from shockbench_flow.policies.mpc_det import MpcDet, MpcDetParams
from shockbench_flow.policies.naive_fq import fallback_spec
from shockbench_flow.policies.registry import PolicyContext


FQ_REPLICATIONS = 1000  # naive's F_Q replications, as the scorer uses them
CACHE = Path.home() / ".cache/shockbench-flow"


@dataclass(frozen=True)
class World:
    """One episode: the instance, its scenario, the scenario's weekly marks, naive's fallback and the policy context."""

    task: str
    entropy: int
    episode: int
    inst: object
    omega: object
    marks: object
    fallback: object
    context: PolicyContext


def world(task: str, entropy: int, episode: int) -> World:
    """Episode ``episode`` of the public generator on root ``entropy``, built as the scorer builds it."""
    inst, params = task_generator(task)
    fq = fq_quantiles(inst, params, FQ_REPLICATIONS, cache_dir=str(CACHE))  # disk hit after the first call
    omega = scenario(task, episode, entropy=entropy)
    return World(
        task, entropy, episode, inst, omega, compute_marks(inst, omega),
        fallback_spec(inst, params, FQ_REPLICATIONS), PolicyContext(fq_quantile=fq.quantiles),
    )


class Teacher(MpcDet):
    """``mpc_det`` with the true future as its window and the whole remaining horizon."""

    def __init__(self, true_marks, context: PolicyContext, rules: bool = True) -> None:
        super().__init__(MpcDetParams(planning_rules=rules), context)
        self.true_marks = true_marks
        self.name = f"teacher[rules={rules}]"
        self.failed_weeks = 0

    def _horizon(self, inst, plan) -> int:
        return inst.T

    def _window_arrays(self, inst, obs: dict, H: int) -> dict:
        t = int(obs["week"])
        return {f: getattr(self.true_marks, f)[t - 1 : t - 1 + H].copy() for f in L.WINDOW_FIELDS}

    def act(self, obs: dict) -> dict:
        inst, t = self._inst, int(obs["week"])
        self._memory.update(inst, obs)
        H = inst.T - t + 1
        arrays = self._window_arrays(inst, obs, H)
        hits = tuple(
            M.FabHit(fab=h.fab, onset=h.onset - (t - 1), severity=h.severity)
            for h in self.true_marks.fab_hits
            if h.onset > t - 1 and t <= h.onset_week <= t + H - 1
        )
        model = L.rolled_lp(inst, obs, arrays, H, fab_hits=hits, planning_rules=self.params.planning_rules)
        res = self._session.solve(L.to_highs_lp(model), L.WindowShape.of(model))
        if not res.ok:
            self.failed_weeks += 1
            return self._fallback.act(obs)
        return L.week1_action(inst, model, res.x, obs, L.prohibited_now(self._memory, t))


def action_arrays(action: dict, static: dict, layout: dict) -> dict[str, np.ndarray]:
    """A protocol action (``flows``, ``overrides``, ``hold`` as slot lists) as the submission's three arrays.

    ``release_mode`` is 2 on a held (chokepoint, commodity) pair, 1 on a pair with any override slot listed (a 0
    quantity included: an override of 0 releases nothing), else 0, the default release.
    """
    n_slots = len(static["action_slots"]["edge"])
    ov = static["override_slots"]
    pairs = [tuple(int(v) for v in p) for p in layout["release_pairs"]]
    flows = np.zeros(n_slots)
    override_qty = np.zeros(len(ov["chokepoint"]))
    release_mode = np.zeros(len(pairs), dtype=np.int64)
    if action is None:
        return {"flows": flows, "override_qty": override_qty, "release_mode": release_mode}
    f = action.get("flows") or {"slot": [], "qty": []}
    flows[np.asarray(f["slot"], dtype=int)] = np.asarray(f["qty"], dtype=float)
    o = action.get("overrides")
    if o:
        override_qty[np.asarray(o["slot"], dtype=int)] = np.asarray(o["qty"], dtype=float)
        for s in o["slot"]:
            release_mode[pairs.index((int(ov["chokepoint"][s]), int(ov["k"][s])))] = 1
    h = action.get("hold")
    if h:
        for c, k in zip(h["chokepoint"], h["k"]):
            release_mode[pairs.index((int(c), int(k)))] = 2
    return {"flows": flows, "override_qty": override_qty, "release_mode": release_mode}
