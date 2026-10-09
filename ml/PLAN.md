# Imitation learning: distil a perfect-foresight planner into a network

## Goal

A submission whose `act` is a small neural network trained to act like a planner that knows the future, then
possibly fine-tuned with RL on the real simulator. Target: above the LP planners (`mpc_det` 0.73, `mpc_scen` 0.75
on Small dev), first towards the teacher's 0.80, then past it with RL.

## What we know (Small, 20 dev episodes, measured 2026-10-09)

| policy | RSS | knows the future |
| --- | --- | --- |
| our best evolved LP (`69ab9689d422`) | ~0.71 (unweighted) | no |
| `mpc_det` | 0.73 | no |
| `mpc_scen` | 0.75 | no (samples futures) |
| perfect planner (the teacher below) | 0.80 | yes |
| board leaders, converted to dev with our +0.03-0.04 offset | ~0.85 | no |

- Knowing the future is worth ~0.07: most of the loss is in how agents act, not in what they know.
- Cost is ~99% chip shortage + power shed. A rule-respecting (base-load-first) clairvoyant MILP scored ~0.997
  on one episode, so the real limit is far above 0.80.

## Hard constraints

- **The private board is Full.** It re-scores the same submission on 400 Full episodes (4 s CPU per week, 480 s
  per episode). A network with Small's input/output sizes crashes on Full, and every crashed week is played by the
  naive rule (RSS ~0). The submission must either ship one model per network (pick by `config["static"]["instance_id"]`
  or by shapes) or use an architecture that does not depend on the shapes.
- **`agent.py` imports only numpy, scipy, torch (CPU, one thread) and the standard library.** Feature extraction
  must be plain numpy inside the agent folder; `ml/` training code may use anything.
- **Never train or tune on dev** (root 0). Train on our own roots, confirm on dev with `sbf compare`.

## The teacher

The perfect planner: `mpc_det` re-solved every week from the **current** state on the **true** future marks
(full remaining horizon, planning rules on). It is a usable teacher because it answers "what would you do here?"
from any state, including states the student reaches. The one-shot clairvoyant LP (score 1.0) is not: it is one
fixed plan, part of it is not an action (fab starts, energy split, container release), and it assumes relaxed rules.

Prototype: the session scratchpad's `perfect.py` (`PerfectMPC(true_marks, context, rules=True)`), to be moved to
`ml/teacher.py`.

Label at each visited state, converted to the submission's action arrays:
- `flows[s]` = the teacher's flow on action slot s (0 where it sends nothing);
- `override_qty[o]` = its release on override slot o;
- `release_mode[p]` = 2 where it holds pair p, 1 where it releases on any override slot of p, else 0.

## Data loop (DAgger)

One environment per episode in the **standard** regime (what the student sees); the teacher additionally holds that
episode's true marks.

1. **Round 0 (behaviour cloning):** the teacher plays; record (student observation, teacher action) every week.
2. **Train** the student on everything recorded so far.
3. **Round k:** the student plays (or a mix: teacher with probability beta_k, decreasing); the teacher labels every
   state visited; add to the data set; go to 2.

Student observations are produced with the kit's own conversion (`shockbench_flow_agent`, as `AgentShim` does), so
training features equal what the agent sees on the server. Episodes come from our own roots of the public
generator (`ScenarioPool` / `sample_omega`).

## Student

- **Features (numpy, shared with `agent.py`):** this week's graph (capacity, open, prohibited, tariff, kappa,
  supply, fab/grid/OSAT state), stock, backlog, demand forecast, warnings; the padded lists aggregated, not dropped:
  in-transit quantity per (destination node, commodity) by arrival week bucket, queue totals per lot key, fab/OSAT
  WIP by weeks to completion; week and weeks left. Normalised by nominal capacities and demand.
- **Outputs:** flows as a fraction of the slot's nominal capacity, overrides as a fraction of the out-edge's
  capacity (regression), release modes as 3-way classification per pair.
- **Model:** an MLP first. If one model per network is not enough for Full, move to a per-slot / per-node model
  (graph network) whose weights do not depend on the counts.
- **Loss:** cost-weighted regression would be ideal but starts as plain MSE on normalised actions + cross-entropy
  on modes.

## RL fine-tuning (only after the student beats ~0.75)

Start PPO from the student's weights (actor) on the real simulator, small learning rate, with a penalty for moving
far from the teacher's actions early on. This is where it can pass the teacher: the reward is the simulator's real
cost, every rule included.

## Milestones (each with a go / no-go)

0. **Teacher check.** Move the teacher to `ml/teacher.py`; re-measure it on Small dev (expect ~0.80); measure its
   time per week on Small and **Full** (Full's clairvoyant LP takes over a minute per episode, the weekly re-solve
   may be too slow to label many Full episodes). No-go on Full -> plan a Small-trained shape-agnostic model, or a
   cheaper Full teacher (shorter horizon).
1. **Data generator** (`ml/collect.py`): round-0 data, `.npz` per episode under `outputs/ml/<date_time>/`, on a
   root of our own; check that replaying the recorded teacher actions reproduces the teacher's cost.
2. **Features + student** (`ml/features.py`, `ml/train.py`): behaviour cloning on round 0. Go if the student on dev
   beats our current best (0.65 weighted with `sbf evaluate`) and passes `sbf check --task=small`.
3. **DAgger rounds** (`ml/dagger.py`): 3-5 rounds; go on if each round improves dev in `sbf compare`.
4. **Full support:** a model per network or a shape-agnostic one; `sbf check --task=full` must pass without
   fallback weeks.
5. **RL fine-tune** from the student, if 2-3 land at or above ~0.75.

## Files (planned)

```
ml/
  PLAN.md         this file
  teacher.py      the perfect planner (training only, imports shockbench_flow)
  collect.py      runs episodes, records (observation, teacher action)
  features.py     observation -> vector, plain numpy; copied into the agent folder
  train.py        the student
  dagger.py       the rounds
agents/il/        agent.py + features.py + weights (what gets submitted)
```

## Status

- 2026-10-09: `ml/teacher.py` and `ml/collect.py` written. Replaying the recorded action arrays through the kit's
  shim reproduces the teacher's cost to the cent (2 Small episodes). Measured on this machine (8 workers):
  Small ~2 s of wall per episode (7 s on one core, 14 s each under 8-way load), 0.13 MB per episode;
  Full 76 s per episode on one core, 0.51 MB per episode, no failed weeks. Labelling Full is feasible.
  Not yet measured: the teacher's RSS on Full.

## Open questions

- Teacher speed on Full (milestone 0) decides how Full is handled.
- How much data: start with ~200 Small episodes (~10k weeks) for round 0 and measure the learning curve.
- Whether to weight samples by episode harm level the way the score does (levels 3-4 are 20% of the score but rare).
