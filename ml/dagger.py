"""DAgger: the student plays, the teacher labels every state the student reaches, the student is retrained on all.

    uv run python -m ml.dagger --agent=il_small --rounds=3 --episodes=200 --n_jobs=8

Round k (1..rounds): episodes of our own root (never dev) are played by a mix, each week the student's action with
probability 1 - beta_k and the teacher's with beta_k (beta_k = beta * decay**(k-1)); the teacher's action is recorded
as the label every week. The weeks go to ``outputs/ml/dagger/<date_time>/round_<k>/`` in ``ml.collect``'s format,
the student is retrained on round 0 plus every round so far (``ml.train``), exported as ``agents/<agent>_d<k>/``, and
scored on the dev episodes for the report (never used for any choice here).
"""

import datetime
import json
import shutil
import time
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed
from shockbench_flow.dynamics.env import rollout
from shockbench_flow_agent.shim import AgentShim, load_agent_class

from ml import train
from ml.collect import POLICY_SEED, REGIME
from ml.teacher import Teacher, action_arrays, world


ROUND_STRIDE = 10_000  # round k plays episodes [k * ROUND_STRIDE, ...) of the root: no episode is used twice


class MixPlays:
    """Each week: the student's action with probability 1 - beta, else the teacher's; the teacher's is the label."""

    name = "dagger_mix"

    def __init__(self, student_cls, teacher: Teacher, beta: float, seed: int) -> None:
        self.teacher, self.beta = teacher, beta
        self.rng = np.random.default_rng(seed)
        self.seen = {}

        class Recording:  # the student, keeping each week's observation as it sees it
            def __init__(rec, config):
                rec.inner = student_cls(config)
                self.config = config

            def act(rec, observation):
                self.seen = {k: np.array(v, copy=True) for k, v in observation.items()}
                return rec.inner.act(observation)

        self.shim = AgentShim(Recording)
        self.weeks: list[tuple[dict, dict]] = []
        self.student_weeks = 0

    def reset(self, static: dict, obs: dict, policy_seed: int) -> None:
        self.shim.reset(static, obs, policy_seed)
        self.teacher.reset(static, obs, policy_seed)
        self.static = static

    def act(self, obs: dict) -> dict:
        student = self.shim.act(obs)  # also records the observation
        teacher = self.teacher.act(obs)
        self.weeks.append((self.seen, action_arrays(teacher, self.static, self.config["layout"])))
        if student is not None and self.rng.random() >= self.beta:
            self.student_weeks += 1
            return student
        return teacher


def play_episode(agent_dir: str, task: str, entropy: int, episode: int, beta: float, out: str) -> dict:
    start = time.perf_counter()
    student_cls = load_agent_class(Path(agent_dir) / "agent.py", f"student_{Path(agent_dir).name}")
    w = world(task, entropy, episode)
    teacher = Teacher(w.marks, w.context)
    policy = MixPlays(student_cls, teacher, beta, seed=episode)
    tr = rollout(w.inst, policy, w.omega, REGIME, POLICY_SEED, marks=w.marks, fallback=w.fallback)
    obs = {f"obs/{k}": np.stack([o[k] for o, _ in policy.weeks]) for k in policy.weeks[0][0]}
    act = {f"act/{k}": np.stack([a[k] for _, a in policy.weeks]) for k in policy.weeks[0][1]}
    np.savez_compressed(Path(out) / f"{task}_{entropy}_{episode:05d}.npz", **obs, **act, J_cents=tr.J_cents,
                        failed_weeks=teacher.failed_weeks)
    if not (Path(out) / "config.json").exists():
        (Path(out) / "config.json").write_text(json.dumps(policy.config, default=lambda x: np.asarray(x).tolist()))
    row = {"episode": episode, "J_played_usd": tr.J_cents / 100, "student_weeks": policy.student_weeks,
           "weeks": len(policy.weeks), "teacher_failed_weeks": teacher.failed_weeks,
           "seconds": round(time.perf_counter() - start, 2)}
    print(json.dumps(row), flush=True)
    return row


def dev_score(agent_dir: str, task: str) -> float:
    from sbf_starter import scoring

    return float(scoring.episode_set(task, "dev", n_jobs=8, verbose=False).score(agent_dir).rss)


def main(agent: str = "il_small", base_data: str = "outputs/ml/collect/small_r0", task: str = "small",
         entropy: int = 20261009, rounds: int = 3, episodes: int = 200, beta: float = 0.5, decay: float = 0.5,
         n_jobs: int = 8, epochs: int = 15, hidden: int = 512, layers: int = 2, score: bool = True):
    run = Path(f"outputs/ml/dagger/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    run.mkdir(parents=True, exist_ok=True)
    data, current, report = [base_data], f"agents/{agent}", []
    if score:
        report.append({"round": 0, "agent": current, "dev_rss": dev_score(current, task)})
        print(json.dumps(report[-1]), flush=True)
    for k in range(1, rounds + 1):
        t0 = time.perf_counter()
        out = run / f"round_{k}"
        out.mkdir()
        b = beta * decay ** (k - 1)
        ns = range(k * ROUND_STRIDE, k * ROUND_STRIDE + episodes)
        rows = Parallel(n_jobs=n_jobs)(delayed(play_episode)(current, task, entropy, n, b, str(out)) for n in ns)
        data.append(str(out))
        name = f"{agent}_d{k}"
        train.main(data=",".join(data), name=name, epochs=epochs, hidden=hidden, layers=layers)
        current = f"agents/{name}"
        entry = {"round": k, "agent": current, "beta": b, "episodes": episodes,
                 "student_share": round(sum(r["student_weeks"] for r in rows) / sum(r["weeks"] for r in rows), 3),
                 "mean_played_cost_usd_bn": round(np.mean([r["J_played_usd"] for r in rows]) / 1e9, 1),
                 "minutes": round((time.perf_counter() - t0) / 60, 1)}
        if score:
            entry["dev_rss"] = dev_score(current, task)
        report.append(entry)
        print(json.dumps(entry), flush=True)
        (run / "report.json").write_text(json.dumps(report, indent=1))
    return str(run)


if __name__ == "__main__":
    fire.Fire(main)
