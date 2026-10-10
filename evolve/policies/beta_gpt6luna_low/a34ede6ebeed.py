# -2.1086730513965266
import numpy as np


class Agent:
    def __init__(self, config=None):
        self.static = config["static"]
        self.layout = config["layout"]
        self.edges = self.static["edges"]
        self.slots = self.static["action_slots"]
        self.nnode = len(self.static["nodes"]["id"])
        self.nk = len(self.static["commodities"]["id"])
        self.nslot = len(self.slots["edge"])
        action = config["spaces"]["action"]
        self.override_qty = np.zeros(action["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action["release_mode"]["shape"], dtype=np.int64)

    @staticmethod
    def _num(x, default=0.0):
        try:
            y = float(x)
            return y if np.isfinite(y) else default
        except (TypeError, ValueError):
            return default

    def act(self, obs):
        flows = np.zeros(self.nslot, dtype=float)
        tails, heads = self.edges["tail"], self.edges["head"]
        nedge = len(tails)
        lane_edges = self.static.get("lanes", {}).get("edges", [])

        stock = np.zeros((self.nnode, self.nk), dtype=float)
        stock_values = np.asarray(obs.get("stock.qty", [])).ravel()
        for i, pair in enumerate(self.layout.get("stock_slots", [])):
            if i < len(stock_values):
                node, k = map(int, pair)
                stock[node, k] += max(0.0, self._num(stock_values[i]))

        supply = np.zeros_like(stock)
        supply_values = np.asarray(obs.get("graph_now.supply.avail", [])).ravel()
        for i, pair in enumerate(self.layout.get("supply_slots", [])):
            if i < len(supply_values):
                node, k = map(int, pair)
                supply[node, k] += max(0.0, self._num(supply_values[i]))

        week_arr = np.asarray(obs.get("week", [1])).ravel()
        week = int(week_arr[0]) if len(week_arr) else 1
        sinks = self.static.get("sinks", {})
        sink_nodes = [int(x) for x in sinks.get("node", [])]
        sink_ks = [int(x) for x in sinks.get("k", [])]
        penalties = sinks.get("pi", [1.0] * len(sink_nodes))
        backlog = np.asarray(obs.get("backlog.qty", [])).ravel()
        forecast = np.asarray(obs.get("demand_forecast.qty", []), dtype=float)

        # The forecast includes the current and following weeks. Avoid building
        # a large speculative stockpile, especially late in the episode.
        need = {}
        for i, (node, k) in enumerate(zip(sink_nodes, sink_ks)):
            b = max(0.0, self._num(backlog[i])) if i < len(backlog) else 0.0
            horizon = min(2, forecast.shape[1]) if forecast.ndim == 2 and i < forecast.shape[0] else 0
            f = sum(max(0.0, self._num(x)) for x in forecast[i, :horizon]) if horizon else 0.0
            need[(node, k)] = max(0.0, b + f - stock[node, k])

        # Credit only live shipments expected to reach the sink within the
        # replenishment window.
        pe = np.asarray(obs.get("pipeline.edge", [])).ravel()
        pk = np.asarray(obs.get("pipeline.k", [])).ravel()
        pq = np.asarray(obs.get("pipeline.qty", [])).ravel()
        pa = np.asarray(obs.get("pipeline.arrival_week", [])).ravel()
        live = np.asarray(obs.get("pipeline.qty.observed", np.ones(len(pq)))).ravel()
        for i in range(min(len(pe), len(pk), len(pq), len(pa))):
            if i < len(live) and not live[i]:
                continue
            e, k = int(pe[i]), int(pk[i])
            if 0 <= e < nedge and 0 <= k < self.nk and int(pa[i]) <= week + 2:
                key = (int(heads[e]), k)
                if key in need:
                    need[key] = max(0.0, need[key] - max(0.0, self._num(pq[i])))

        cap_now = np.asarray(obs.get("graph_now.u", [])).ravel()
        cost_now = np.asarray(obs.get("graph_now.c", [])).ravel()
        tau_now = np.asarray(obs.get("graph_now.tau", [])).ravel()
        mask = np.asarray(obs.get("action_mask", np.ones(self.nslot))).ravel()
        mask_observed = bool(np.asarray(obs.get("action_mask.observed", [1])).ravel()[0])
        prohibited = np.asarray(obs.get("slot_mask", np.zeros(self.nslot))).ravel()
        edge_left = np.full(nedge, np.inf)
        for e in range(min(nedge, len(cap_now))):
            if np.isfinite(cap_now[e]):
                edge_left[e] = max(0.0, float(cap_now[e]))

        slot_edges = self.slots["edge"]
        slot_ks = self.slots["k"]
        slot_lanes = self.slots.get("lane", [None] * self.nslot)

        for k in range(self.nk):
            arcs = []
            for s in range(self.nslot):
                if int(slot_ks[s]) != k:
                    continue
                allowed = (s < len(mask) and mask[s] > 0.5) if mask_observed else not (s < len(prohibited) and prohibited[s] > 0.5)
                if not allowed:
                    continue
                route = [int(slot_edges[s])]
                lane = slot_lanes[s] if s < len(slot_lanes) else None
                if lane is not None:
                    try:
                        candidate = lane_edges[int(lane)]
                        if candidate:
                            route = [int(e) for e in candidate]
                    except (TypeError, ValueError, IndexError):
                        pass
                if not route or any(e < 0 or e >= nedge for e in route):
                    continue
                cap = min(edge_left[e] for e in route)
                cost = 0.0
                tau = 0.0
                for e in route:
                    if e < len(cap_now) and np.isfinite(cap_now[e]):
                        cap = min(cap, max(0.0, float(cap_now[e])))
                    if e < len(cost_now):
                        cost += max(0.0, self._num(cost_now[e]))
                    if e < len(tau_now):
                        tau += max(0.0, self._num(tau_now[e], 1.0))
                if cap <= 1e-9:
                    continue
                u, v = int(tails[route[0]]), int(heads[route[-1]])
                arcs.append([s, route, u, v, cap, cost + 0.15 * tau])

            # Recompute shortest paths as capacity is consumed. This allows
            # another route to be selected when the cheapest one fills up.
            for si, (sink, kk) in enumerate(zip(sink_nodes, sink_ks)):
                if kk != k:
                    continue
                remaining = need.get((sink, k), 0.0)
                pi = max(0.0, self._num(penalties[si]) if si < len(penalties) else 1.0)
                while remaining > 1e-8:
                    best = None
                    for origin in range(self.nnode):
                        available = stock[origin, k] + supply[origin, k]
                        if origin == sink or available <= 1e-8:
                            continue
                        # Reverse relaxation finds the cheapest path from each
                        # origin to this sink; retain the first action slot.
                        dist = [float("inf")] * self.nnode
                        first = [-1] * self.nnode
                        dist[sink] = 0.0
                        for _ in range(max(0, self.nnode - 1)):
                            changed = False
                            for ai, a in enumerate(arcs):
                                if a[4] <= 1e-9 or not np.isfinite(dist[a[3]]):
                                    continue
                                nd = a[5] + dist[a[3]]
                                if nd < dist[a[2]] - 1e-10:
                                    dist[a[2]] = nd
                                    first[a[2]] = ai
                                    changed = True
                            if not changed:
                                break
                        ai = first[origin]
                        if ai < 0 or not np.isfinite(dist[origin]):
                            continue
                        a = arcs[ai]
                        qty_cap = min(a[4], *(edge_left[e] for e in a[1]))
                        if qty_cap <= 1e-9:
                            continue
                        candidate = (dist[origin] / max(pi, 1e-9), dist[origin], origin, ai, available, qty_cap)
                        if best is None or candidate[:3] < best[:3]:
                            best = candidate
                    if best is None:
                        break
                    _, _, origin, ai, available, qty_cap = best
                    a = arcs[ai]
                    qty = min(remaining, available, qty_cap)
                    if qty <= 1e-9:
                        break
                    flows[a[0]] += qty
                    used_stock = min(qty, stock[origin, k])
                    stock[origin, k] -= used_stock
                    supply[origin, k] = max(0.0, supply[origin, k] - (qty - used_stock))
                    for e in a[1]:
                        edge_left[e] = max(0.0, edge_left[e] - qty)
                    a[4] = max(0.0, a[4] - qty)
                    remaining -= qty
                need[(sink, k)] = remaining

        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}