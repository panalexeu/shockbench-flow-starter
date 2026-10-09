"""Record (observation, teacher action) for every week of episodes of our own root: DAgger's round 0.

    uv run python -m ml.collect                                   # 2 Small episodes, root 20261009
    uv run python -m ml.collect --episodes=200 --n_jobs=8         # a round-0 data set
    uv run python -m ml.collect --episodes=2 --verify             # replay the recorded actions, compare the cost

The observation is the dict a submission's ``act`` receives (the kit's own conversion), so a student trained on it
sees on the server exactly what it saw in training. One ``.npz`` per episode under ``outputs/ml/collect/<date_time>/``:
``obs/<key>`` stacked over weeks, ``act/flows``, ``act/override_qty``, ``act/release_mode``, and the teacher's
cost; ``config.json`` holds the agent config of the task (the same for every episode).
"""

import datetime
import json
import time
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed
from shockbench_flow.dynamics.env import rollout
from shockbench_flow_agent.shim import AgentShim

from ml.teacher import Teacher, action_arrays, world


REGIME = "standard"  # what the student sees (the board's regime); the teacher reads the true future regardless
POLICY_SEED = 0


class _Recorder:
    """A submission ``Agent`` that only keeps what it is given: the config and each week's observation."""

    config = None
    observation = None

    def __init__(self, config):
        _Recorder.config = config

    def act(self, observation):
        _Recorder.observation = {k: np.array(v, copy=True) for k, v in observation.items()}
        return {"flows": np.zeros(len(_Recorder.config["static"]["action_slots"]["edge"]))}


class _Replay:
    """A submission ``Agent`` that plays recorded action arrays, week by week."""

    actions = None

    def __init__(self, config):
        self.week = 0

    def act(self, observation):
        a = {k: v[self.week] for k, v in _Replay.actions.items()}
        self.week += 1
        return a


class TeacherPlays:
    """The teacher plays; the shim records the student's view of each state and the teacher's action as arrays."""

    name = "teacher_plays"

    def __init__(self, teacher: Teacher) -> None:
        self.teacher = teacher
        self.shim = AgentShim(_Recorder)
        self.weeks: list[tuple[dict, dict]] = []

    def reset(self, static: dict, obs: dict, policy_seed: int) -> None:
        self.shim.reset(static, obs, policy_seed)
        self.teacher.reset(static, obs, policy_seed)
        self.static, self.layout = static, _Recorder.config["layout"]

    def act(self, obs: dict) -> dict:
        self.shim.act(obs)  # the student's view of this state
        action = self.teacher.act(obs)
        self.weeks.append((_Recorder.observation, action_arrays(action, self.static, self.layout)))
        return action


def collect_episode(task: str, entropy: int, episode: int, out: str) -> dict:
    start = time.perf_counter()
    w = world(task, entropy, episode)
    teacher = Teacher(w.marks, w.context)
    policy = TeacherPlays(teacher)
    tr = rollout(w.inst, policy, w.omega, REGIME, POLICY_SEED, marks=w.marks, fallback=w.fallback)
    obs = {f"obs/{k}": np.stack([o[k] for o, _ in policy.weeks]) for k in policy.weeks[0][0]}
    act = {f"act/{k}": np.stack([a[k] for _, a in policy.weeks]) for k in policy.weeks[0][1]}
    path = Path(out) / f"{task}_{entropy}_{episode:05d}.npz"
    np.savez_compressed(path, **obs, **act, J_cents=tr.J_cents, failed_weeks=teacher.failed_weeks)
    row = {
        "episode": episode, "J_teacher_usd": tr.J_cents / 100, "weeks": len(policy.weeks),
        "teacher_failed_weeks": teacher.failed_weeks, "seconds": round(time.perf_counter() - start, 2),
        "mb": round(path.stat().st_size / 2**20, 2),
    }
    if episode == 0 or not (Path(out) / "config.json").exists():
        (Path(out) / "config.json").write_text(json.dumps(_Recorder.config, default=lambda x: np.asarray(x).tolist()))
    print(json.dumps(row), flush=True)
    return row


def verify_episode(path: Path, task: str, entropy: int, episode: int) -> dict:
    """Replay the recorded action arrays through the kit's shim; the cost must equal the teacher's."""
    d = np.load(path)
    _Replay.actions = {k.removeprefix("act/"): d[k] for k in d.files if k.startswith("act/")}
    w = world(task, entropy, episode)
    tr = rollout(w.inst, AgentShim(_Replay), w.omega, REGIME, POLICY_SEED, marks=w.marks, fallback=w.fallback)
    return {"episode": episode, "J_teacher_cents": int(d["J_cents"]), "J_replay_cents": tr.J_cents,
            "equal": int(d["J_cents"]) == tr.J_cents}


def main(task: str = "small", entropy: int = 20261009, episodes: int = 2, start: int = 0, n_jobs: int = 1,
         verify: bool = False, out: str | None = None):
    out = out or f"outputs/ml/collect/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    Path(out).mkdir(parents=True, exist_ok=True)
    ns = list(range(start, start + episodes))
    t0 = time.perf_counter()
    rows = Parallel(n_jobs=n_jobs)(delayed(collect_episode)(task, entropy, n, out) for n in ns)
    wall = time.perf_counter() - t0
    summary = {
        "task": task, "entropy": entropy, "episodes": ns, "n_jobs": n_jobs, "wall_seconds": round(wall, 1),
        "seconds_per_episode": round(sum(r["seconds"] for r in rows) / len(rows), 2),
        "mb_per_episode": round(sum(r["mb"] for r in rows) / len(rows), 2), "rows": rows,
    }
    if verify:
        summary["verify"] = [verify_episode(Path(out) / f"{task}_{entropy}_{n:05d}.npz", task, entropy, n) for n in ns]
    (Path(out) / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    return out


if __name__ == "__main__":
    fire.Fire(main)
