# -2.1076266577983516
import numpy as np


class Agent:
    def __init__(self, config=None):
        self.static = config["static"]
        self.layout = config["layout"]
        self.T = int(config.get("T", self.static.get("T", 52)))
        self.edges = self.static["edges"]
        self.slots = self.static["action_slots"]
        self.stock_slots = self.layout["stock_slots"]
        self.supply_slots = self.layout["supply_slots"]
        self.demands = self.layout["demands"]
        self.nodes = len(self.static["nodes"]["id"])
        self.nk = len(self.static["commodities"]["id"])
        self.nslot = len(self.slots["edge"])
        self.override_qty = np.zeros(config["spaces"]["action"]["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(config["spaces"]["action"]["release_mode"]["shape"], dtype=np.int64)
        self.slot_by_edge_k = {}
        for s, (e, k) in enumerate(zip(self.slots["edge"], self.slots["k"])):
            self.slot_by_edge_k.setdefault((int(e), int(k)), []).append(s)

    @staticmethod
    def _finite(x, default=0.0):
        try:
            y = float(x)
            return y if np.isfinite(y) else default
        except Exception:
            return default

    def act(self, obs):
        flows = np.zeros(self.nslot, dtype=float)
        week = int(np.asarray(obs.get("week", [1])).ravel()[0])
        edge_tail = self.edges["tail"]
        edge_head = self.edges["head"]
        edge_mode = self.edges["mode"]
        lane_edges = self.static.get("lanes", {}).get("edges", [])
        slot_lane = self.slots.get("lane", [None] * self.nslot)
        slot_edge = [int(x) for x in self.slots["edge"]]
        slot_k = [int(x) for x in self.slots["k"]]
        mask = np.asarray(obs.get("action_mask", np.ones(self.nslot)), dtype=float).ravel()
        prohibited = np.asarray(obs.get("slot_mask", np.zeros(self.nslot)), dtype=float).ravel()
        mask_obs = int(np.asarray(obs.get("action_mask.observed", [1])).ravel()[0])
        if mask_obs == 0:
            mask = np.ones(self.nslot)
        cap_now = np.asarray(obs.get("graph_now.u", []), dtype=float).ravel()
        cost_now = np.asarray(obs.get("graph_now.c", []), dtype=float).ravel()
        tau_now = np.asarray(obs.get("graph_now.tau", []), dtype=float).ravel()
        open_cp = np.asarray(obs.get("graph_now.open", []), dtype=float).ravel()
        cp_nodes = [int(x) for x in self.layout.get("chokepoints", [])]
        cp_index = {n: i for i, n in enumerate(cp_nodes)}
        stock = np.zeros((self.nodes, self.nk), dtype=float)
        for i, pair in enumerate(self.stock_slots):
            if i < len(np.asarray(obs.get("stock.qty", [])).ravel()):
                n, k = map(int, pair)
                stock[n, k] += max(0.0, self._finite(obs["stock.qty"][i]))
        supply = np.zeros((self.nodes, self.nk), dtype=float)
        av = np.asarray(obs.get("graph_now.supply.avail", []), dtype=float).ravel()
        for i, pair in enumerate(self.supply_slots):
            if i < len(av):
                n, k = map(int, pair)
                supply[n, k] += max(0.0, self._finite(av[i]))
        # Count shipments already destined for each node and commodity, to avoid
        # repeatedly ordering cargo that is already in transit.
        incoming = np.zeros((self.nodes, self.nk), dtype=float)
        pe = np.asarray(obs.get("pipeline.edge", [])).ravel()
        pk = np.asarray(obs.get("pipeline.k", [])).ravel()
        pq = np.asarray(obs.get("pipeline.qty", [])).ravel()
        pa = np.asarray(obs.get("pipeline.arrival_week", [])).ravel()
        for i in range(min(len(pe), len(pk), len(pq), len(pa))):
            e, k = int(pe[i]), int(pk[i])
            if 0 <= e < len(edge_head) and 0 <= k < self.nk and int(pa[i]) <= week + 3:
                incoming[int(edge_head[e]), k] += max(0.0, self._finite(pq[i]))
        backlog = np.asarray(obs.get("backlog.qty", []), dtype=float).ravel()
        forecast = np.asarray(obs.get("demand_forecast.qty", []), dtype=float)
        sink_nodes = [int(x) for x in self.static.get("sinks", {}).get("node", [])]
        sink_ks = [int(x) for x in self.static.get("sinks", {}).get("k", [])]
        sink_pi = self.static.get("sinks", {}).get("pi", [1.0] * len(sink_nodes))
        sink_need = {}
        for i, (n, k) in enumerate(zip(sink_nodes, sink_ks)):
            f = 0.0
            if forecast.ndim == 2 and i < forecast.shape[0]:
                f = sum(max(0.0, self._finite(x)) for x in forecast[i, :min(forecast.shape[1], 3)])
            b = max(0.0, self._finite(backlog[i])) if i < len(backlog) else 0.0
            sink_need[(n, k)] = max(0.0, b + f - stock[n, k] - incoming[n, k])

        # Construct commodity-specific routes over action-slot arcs. A lane slot
        # represents the whole lane; its destination is the lane's last edge head.
        for k in range(self.nk):
            arcs = []
            for s in range(self.nslot):
                if slot_k[s] != k or (mask_obs and s < len(mask) and mask[s] < 0.5) or (not mask_obs and s < len(prohibited) and prohibited[s] > 0.5):
                    continue
                e = slot_edge[s]
                lane = slot_lane[s] if s < len(slot_lane) else None
                route = [e]
                if lane is not None:
                    try:
                        li = int(lane)
                        if 0 <= li < len(lane_edges) and lane_edges[li]:
                            route = [int(x) for x in lane_edges[li]]
                    except Exception:
                        pass
                if not route:
                    continue
                start, end = int(edge_tail[route[0]]), int(edge_head[route[-1]])
                cap = float('inf')
                cost = 0.0
                tau = 0.0
                for re in route:
                    if re < len(cap_now) and np.isfinite(cap_now[re]):
                        cap = min(cap, max(0.0, cap_now[re]))
                    if re < len(cost_now):
                        cost += max(0.0, self._finite(cost_now[re]))
                    elif re < len(self.edges.get("c0", [])):
                        cost += max(0.0, self._finite(self.edges["c0"][re]))
                    if re < len(tau_now):
                        tau += max(0.0, self._finite(tau_now[re]))
                    else:
                        tau += max(0.0, self._finite(self.edges.get("tau0", [1] * len(edge_tail))[re], 1))
                    cp = cp_index.get(int(edge_head[re]))
                    if cp is not None and cp < len(open_cp):
                        cap *= max(0.0, min(1.0, self._finite(open_cp[cp])))
                if not np.isfinite(cap):
                    cap = 1e12
                cap = min(cap, 1e12)
                arcs.append((s, start, end, cap, cost + 0.1 * tau, tau))
            # Floyd-Warshall distances and first arc index.
            inf = 1e15
            dist = np.full((self.nodes, self.nodes), inf)
            first = np.full((self.nodes, self.nodes), -1, dtype=int)
            for n in range(self.nodes):
                dist[n, n] = 0.0
            for ai, (_, u, v, cap, weight, _) in enumerate(arcs):
                if cap > 0 and weight < dist[u, v]:
                    dist[u, v] = weight
                    first[u, v] = ai
            for mid in range(self.nodes):
                for u in range(self.nodes):
                    if dist[u, mid] >= inf:
                        continue
                    cand = dist[u, mid] + dist[mid]
                    better = cand < dist[u] - 1e-9
                    dist[u, better] = cand[better]
                    first[u, better] = first[u, mid]
            # Pull each sink's need back to reachable inventory/supply. First
            # allocate directly from closest candidate origins to avoid waste.
            remaining = {n: need for (n, kk), need in sink_need.items() if kk == k and need > 1e-9}
            for sink, need in sorted(remaining.items(), key=lambda x: -x[1]):
                origins = sorted(range(self.nodes), key=lambda n: dist[n, sink])
                for origin in origins:
                    if need <= 1e-8 or origin == sink or dist[origin, sink] >= inf:
                        continue
                    ai = int(first[origin, sink])
                    if ai < 0:
                        continue
                    s, u, v, cap, _, _ = arcs[ai]
                    available = stock[u, k] + supply[u, k]
                    if available <= 1e-8 or cap <= 1e-8:
                        continue
                    # Reserve only the amount this immediate leg can move.
                    qty = min(need, available, cap)
                    flows[s] += qty
                    stock[u, k] = max(0.0, stock[u, k] - qty)
                    supply[u, k] = max(0.0, supply[u, k] - qty)
                    need -= qty
                # Do not dispatch unrelated surplus merely to fill capacity.
        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}
