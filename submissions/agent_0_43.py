import numpy as np

class Agent:
    def __init__(self, config=None):
        static = config["static"]
        instance = static.get("instance", {})
        slots = static["action_slots"]
        self.slot_edges = np.asarray(slots["edge"], dtype=int)
        self.slot_k = np.asarray(slots["k"], dtype=int)
        self.slot_lane = np.asarray([-1 if x is None else int(x) for x in slots["lane"]], dtype=int)
        self.capacity = np.asarray([0.0 if x is None else float(x) for x in static["edges"]["u0"]])[self.slot_edges]
        self.nominal = np.zeros(len(self.slot_edges), dtype=float)
        pipe = instance.get("initial_state", {}).get("pipeline", [])
        if isinstance(pipe, dict):
            n = len(pipe.get("edge", []))
            rows = [{key: value[i] for key, value in pipe.items()
                     if isinstance(value, (list, tuple, np.ndarray)) and len(value) > i}
                    for i in range(n)]
        elif isinstance(pipe, (list, tuple)):
            rows = pipe
        else:
            rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                edge = int(row.get("edge", row.get("e")))
                commodity = int(row.get("k", row.get("commodity")))
                lane_value = row.get("lane")
                lane = -1 if lane_value is None else int(lane_value)
                qty = max(0.0, float(row.get("qty", row.get("quantity", 0.0))))
            except (TypeError, ValueError):
                continue
            for s in range(len(self.nominal)):
                if (self.slot_edges[s] == edge and self.slot_k[s] == commodity
                        and self.slot_lane[s] == lane):
                    self.nominal[s] += qty
        if not np.any(self.nominal > 0):
            self.nominal = self.capacity.copy()
        action = config["spaces"]["action"]
        self.override_qty = np.zeros(action["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action["release_mode"]["shape"], dtype=np.int64)
        node_ids = static.get("nodes", {}).get("id", [])
        cp_positions = {}
        for pos, cp in enumerate(config.get("layout", {}).get("chokepoints", [])):
            if isinstance(cp, str):
                cp_positions[cp] = pos
            else:
                try:
                    cp_positions[str(node_ids[int(cp)])] = pos
                except (IndexError, TypeError, ValueError):
                    pass
        self.lane_chokepoints = []
        for lane_data in instance.get("lanes", []):
            cps = lane_data.get("chokepoints", []) if isinstance(lane_data, dict) else []
            mapped = []
            for cp in cps:
                if isinstance(cp, str):
                    pos = cp_positions.get(cp)
                else:
                    try:
                        pos = cp_positions.get(str(node_ids[int(cp)]))
                    except (IndexError, TypeError, ValueError):
                        pos = None
                if pos is not None:
                    mapped.append(pos)
            self.lane_chokepoints.append(mapped)

    def act(self, observation):
        allowed = np.asarray(observation["action_mask"], dtype=float)
        capacity = np.asarray(observation["graph_now.u"], dtype=float)
        open_fraction = np.asarray(observation["graph_now.open"], dtype=float)
        week = int(np.asarray(observation["week"]).reshape(-1)[0])
        pending_edge = np.asarray(observation["pending_prohibitions.edge"])
        pending_k = np.asarray(observation["pending_prohibitions.k"])
        pending_week = np.asarray(observation["pending_prohibitions.effective_week"])
        pending_live = np.asarray(observation["pending_prohibitions.edge.observed"], dtype=bool)
        flows = np.zeros_like(self.nominal)
        for s, (edge, commodity, lane) in enumerate(zip(self.slot_edges, self.slot_k, self.slot_lane)):
            if allowed[s] <= 0:
                continue
            target = self.nominal[s]
            # Modestly stock up only when a matching prohibition takes effect next week.
            soon_banned = np.any(pending_live & (pending_edge == edge) & (pending_k == commodity)
                                 & (pending_week == week + 1))
            if soon_banned:
                target *= 1.25
            qty = min(target, max(0.0, capacity[edge]))
            if lane >= 0 and lane < len(self.lane_chokepoints):
                cps = self.lane_chokepoints[lane]
                if cps:
                    if any(cp >= len(open_fraction) for cp in cps):
                        qty = 0.0
                    else:
                        qty *= min(float(np.clip(open_fraction[cp], 0.0, 1.0)) for cp in cps)
            flows[s] = qty
        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}