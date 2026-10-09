"""Observation -> feature vector, and action <-> normalised targets, in plain numpy (it also runs inside agent.py).

Built once per episode from ``config``; every size is read from it, so the same code serves Tiny, Small and Full
(a trained network still fits only the sizes it was trained on).

The padded lists are aggregated, not dropped, since they hold the inventory in motion:
- cargo in transit per (destination stock slot, arrival bucket), its destination the end of its lane;
- strait queues per lot key;
- fab and OSAT work in process per (stock slot, completion bucket);
- weeks until an announced prohibition hits each action slot's route, and until each closure's announced end.
"""

import numpy as np


ARRIVAL_BUCKETS = (0, 1, 2, 3, 5, 9)  # weeks from now: [0], [1], [2], [3, 4], [5, 8], [9, inf)
WIP_BUCKETS = (0, 2, 4, 9)            # weeks to completion: [0, 1], [2, 3], [4, 8], [9, inf)
HORIZON_CAP = 20                      # weeks: "not announced" and "far away" both read as this


def _bucket(dt: np.ndarray, edges: tuple[int, ...]) -> np.ndarray:
    return np.searchsorted(np.asarray(edges), dt, side="right") - 1


class Features:
    """``vector(observation)`` for one episode's config; ``targets`` / ``action`` map actions to and from [0, ~1]."""

    def __init__(self, config: dict) -> None:
        s, lay = config["static"], config["layout"]
        self.T = int(config["T"])
        ed = s["edges"]
        self.head = np.asarray(ed["head"], dtype=int)
        self.u0 = np.asarray([0.0 if x is None else float(x) for x in ed["u0"]])
        self.lanes = [list(map(int, x)) for x in s["lanes"]["edges"]]
        self.slot_index = {(int(n), int(k)): j for j, (n, k) in enumerate(lay["stock_slots"])}
        self.S = len(lay["stock_slots"])
        self.cp_index = {int(n): j for j, n in enumerate(lay["chokepoints"])}
        sl = s["action_slots"]
        self.slot_edge = np.asarray(sl["edge"], dtype=int)
        self.slot_k = np.asarray(sl["k"], dtype=int)
        self.slot_path = [self.lanes[ln] if ln is not None else [int(e)] for e, ln in zip(sl["edge"], sl["lane"])]
        ov = s["override_slots"]
        self.ov_edge = np.asarray(ov["out_edge"], dtype=int)
        self.n_slots, self.n_ov, self.n_pairs = len(self.slot_edge), len(self.ov_edge), len(lay["release_pairs"])
        # the scales of the targets: a slot's (override slot's) nominal capacity, on its first (out) edge
        self.flow_scale = np.maximum(self.u0[self.slot_edge], 1.0)
        self.ov_scale = np.maximum(self.u0[self.ov_edge], 1.0)

    # ----- observation -> vector -----------------------------------------------------------------------------------
    def _dest(self, edge: int, lane: int | None) -> int:
        if lane is not None and 0 <= lane < len(self.lanes):
            return int(self.head[self.lanes[lane][-1]])
        return int(self.head[edge])

    def _transit(self, o: dict, week: int) -> np.ndarray:
        out = np.zeros((self.S, len(ARRIVAL_BUCKETS)))
        live = np.flatnonzero(o["pipeline.qty.observed"] > 0)
        lane_ok = o["pipeline.lane.observed"]
        for j in live:
            lane = int(o["pipeline.lane"][j]) if lane_ok[j] else None
            slot = self.slot_index.get((self._dest(int(o["pipeline.edge"][j]), lane), int(o["pipeline.k"][j])))
            if slot is not None:
                out[slot, _bucket(int(o["pipeline.arrival_week"][j]) - week, ARRIVAL_BUCKETS)] += o["pipeline.qty"][j]
        return out.ravel()

    def _wip(self, o: dict, week: int) -> np.ndarray:
        out = np.zeros((self.S, len(WIP_BUCKETS)))
        for j in np.flatnonzero(o["wip.qty.observed"] > 0):
            slot = self.slot_index.get((int(o["wip.node"][j]), int(o["wip.k"][j])))
            if slot is not None:
                out[slot, _bucket(max(0, int(o["wip.out_week"][j]) - week), WIP_BUCKETS)] += o["wip.qty"][j]
        return out.ravel()

    def _pending(self, o: dict, week: int) -> np.ndarray:
        soon: dict[tuple[int, int], int] = {}
        for j in np.flatnonzero(o["pending_prohibitions.edge.observed"] > 0):
            key = (int(o["pending_prohibitions.edge"][j]), int(o["pending_prohibitions.k"][j]))
            soon[key] = min(soon.get(key, 10**9), int(o["pending_prohibitions.effective_week"][j]) - week)
        out = np.full(self.n_slots, float(HORIZON_CAP))
        for i, (path, k) in enumerate(zip(self.slot_path, self.slot_k)):
            dts = [soon[(e, int(k))] for e in path if (e, int(k)) in soon]
            if dts:
                out[i] = float(np.clip(min(dts), 0, HORIZON_CAP))
        return out

    def _closure_end(self, o: dict, week: int) -> np.ndarray:
        out = np.full(len(self.cp_index), float(HORIZON_CAP))
        for j in np.flatnonzero(o["closure_end.end_week.observed"] > 0):
            c = self.cp_index.get(int(o["closure_end.chokepoint"][j]))
            if c is not None:
                out[c] = min(out[c], float(np.clip(int(o["closure_end.end_week"][j]) - week, 0, HORIZON_CAP)))
        return out

    def vector(self, o: dict) -> np.ndarray:
        week = int(np.asarray(o["week"]).reshape(-1)[0])
        u = np.asarray(o["graph_now.u"], dtype=float)
        parts = [
            np.array([week / self.T, (self.T - week + 1) / self.T]),
            o["stock.qty"], o["backlog.qty"],
            self._transit(o, week),
            np.asarray(o["queue_lots.qty"], dtype=float).sum(axis=1) if "queue_lots.qty" in o else np.zeros(0),
            self._wip(o, week),
            u / np.maximum(self.u0, 1.0), o["graph_now.u.observed"], o["graph_now.c"], o["graph_now.tau"],
            np.asarray(o["graph_now.prohibited"]).ravel(), np.asarray(o["graph_now.tariff"]).ravel(),
            o["graph_now.open"], o["graph_now.kappa.tb"], o["graph_now.kappa.ct"], o["graph_now.war_risk"],
            o["graph_now.supply.avail"],
            o["graph_now.fab.R"], o["graph_now.fab.alpha_bar"], o["graph_now.fab.cap_eff"],
            o["graph_now.grid.G_bar"], o["graph_now.grid.y_bar"], o["graph_now.osat.R"], o["graph_now.osat.thr_eff"],
            o["action_mask"], o["override_mask"],
            o["last_week.clip.requested"] / self.flow_scale, o["last_week.clip.executed"] / self.flow_scale,
            o["last_week.cost_components"], o["last_week.sinks.demand"], o["last_week.sinks.served"],
            o["last_week.sinks.lost"], o["last_week.shed.qty"],
            np.asarray(o["demand_forecast.qty"]).ravel(), o["warning.score"],
            self._pending(o, week), self._closure_end(o, week),
        ]
        return np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in parts])

    # ----- actions <-> targets -------------------------------------------------------------------------------------
    def targets(self, flows: np.ndarray, override_qty: np.ndarray) -> np.ndarray:
        """Flows and overrides as fractions of nominal capacity, one vector (release modes are classes, apart)."""
        return np.concatenate([flows / self.flow_scale, override_qty / self.ov_scale], axis=-1)

    def action(self, y: np.ndarray, mode_logits: np.ndarray, o: dict) -> dict:
        """A network output back to the submission's action: non-negative, zero where the slot is prohibited."""
        y = np.maximum(np.asarray(y, dtype=float), 0.0)
        flows = y[: self.n_slots] * self.flow_scale * (np.asarray(o["action_mask"]) > 0)
        override_qty = y[self.n_slots :] * self.ov_scale * (np.asarray(o["override_mask"]) > 0)
        release_mode = np.asarray(mode_logits).reshape(self.n_pairs, 3).argmax(axis=1).astype(np.int64)
        return {"flows": flows, "override_qty": override_qty, "release_mode": release_mode}
