# -2.108030827895761
import numpy as np


class Agent:
    def __init__(self, config=None):
        self.static = config["static"]
        self.layout = config["layout"]
        self.edges = self.static["edges"]
        self.slots = self.static["action_slots"]
        self.nn = len(self.static["nodes"]["id"])
        self.nk = len(self.static["commodities"]["id"])
        self.ns = len(self.slots["edge"])
        action = config["spaces"]["action"]
        self.override_qty = np.zeros(action["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action["release_mode"]["shape"], dtype=np.int64)

    @staticmethod
    def num(x, default=0.0):
        try:
            x = float(x)
            return x if np.isfinite(x) else default
        except Exception:
            return default

    def act(self, obs):
        flows = np.zeros(self.ns, dtype=float)
        tails, heads = self.edges["tail"], self.edges["head"]
        lane_edges = self.static.get("lanes", {}).get("edges", [])
        slot_edges = self.slots["edge"]
        slot_ks = self.slots["k"]
        slot_lanes = self.slots.get("lane", [None] * self.ns)
        week = int(np.asarray(obs.get("week", [1])).ravel()[0])

        stock = np.zeros((self.nn, self.nk), dtype=float)
        sq = np.asarray(obs.get("stock.qty", [])).ravel()
        for i, pair in enumerate(self.layout.get("stock_slots", [])):
            if i < len(sq):
                node, k = map(int, pair)
                stock[node, k] += max(0.0, self.num(sq[i]))
        supply = np.zeros_like(stock)
        av = np.asarray(obs.get("graph_now.supply.avail", [])).ravel()
        for i, pair in enumerate(self.layout.get("supply_slots", [])):
            if i < len(av):
                node, k = map(int, pair)
                supply[node, k] += max(0.0, self.num(av[i]))

        sinks = self.static.get("sinks", {})
        snodes = [int(x) for x in sinks.get("node", [])]
        sks = [int(x) for x in sinks.get("k", [])]
        penalties = sinks.get("pi", [1.0] * len(snodes))
        backlog = np.asarray(obs.get("backlog.qty", [])).ravel()
        forecast = np.asarray(obs.get("demand_forecast.qty", []), dtype=float)
        # A small forward buffer helps account for lead times without routinely
        # filling sinks with several weeks of excess inventory.
        need = {}
        for i, (node, k) in enumerate(zip(snodes, sks)):
            f = 0.0
            if forecast.ndim == 2 and i < forecast.shape[0]:
                f = sum(max(0.0, self.num(x)) for x in forecast[i, :min(3, forecast.shape[1])])
            b = max(0.0, self.num(backlog[i])) if i < len(backlog) else 0.0
            need[(node, k)] = max(0.0, b + f - stock[node, k])

        # Credit pipeline only when its final edge reaches the demand sink.
        pe = np.asarray(obs.get("pipeline.edge", [])).ravel()
        pk = np.asarray(obs.get("pipeline.k", [])).ravel()
        pq = np.asarray(obs.get("pipeline.qty", [])).ravel()
        pa = np.asarray(obs.get("pipeline.arrival_week", [])).ravel()
        live = np.asarray(obs.get("pipeline.qty.observed", np.ones(len(pq)))).ravel()
        for i in range(min(len(pe), len(pk), len(pq), len(pa))):
            if i < len(live) and not live[i]:
                continue
            e, k = int(pe[i]), int(pk[i])
            if 0 <= e < len(heads) and 0 <= k < self.nk:
                key = (int(heads[e]), k)
                if key in need and int(pa[i]) <= week + 3:
                    need[key] = max(0.0, need[key] - max(0.0, self.num(pq[i])))

        cap_now = np.asarray(obs.get("graph_now.u", [])).ravel()
        cost_now = np.asarray(obs.get("graph_now.c", [])).ravel()
        tau_now = np.asarray(obs.get("graph_now.tau", [])).ravel()
        opened = np.asarray(obs.get("graph_now.open", [])).ravel()
        cp_pos = {int(node): i for i, node in enumerate(self.layout.get("chokepoints", []))}
        mask = np.asarray(obs.get("action_mask", np.ones(self.ns))).ravel()
        mask_seen = bool(np.asarray(obs.get("action_mask.observed", [1])).ravel()[0])
        prohibited = np.asarray(obs.get("slot_mask", np.zeros(self.ns))).ravel()
        edge_left = np.full(len(tails), np.inf)
        for e in range(len(tails)):
            if e < len(cap_now) and np.isfinite(cap_now[e]):
                edge_left[e] = max(0.0, float(cap_now[e]))

        for k in range(self.nk):
            arcs = []
            for s in range(self.ns):
                if int(slot_ks[s]) != k:
                    continue
                allowed = mask[s] > 0.5 if mask_seen and s < len(mask) else not (s < len(prohibited) and prohibited[s] > 0.5)
                if not allowed:
                    continue
                e = int(slot_edges[s])
                route = [e]
                lane = slot_lanes[s] if s < len(slot_lanes) else None
                if lane is not None:
                    try:
                        le = lane_edges[int(lane)]
                        if le:
                            route = [int(x) for x in le]
                    except Exception:
                        pass
                cap, cost, tau, factor = np.inf, 0.0, 0.0, 1.0
                for re in route:
                    if re < 0 or re >= len(tails):
                        cap = 0.0
                        break
                    if re < len(cap_now) and np.isfinite(cap_now[re]):
                        cap = min(cap, max(0.0, float(cap_now[re])))
                    cost += max(0.0, self.num(cost_now[re])) if re < len(cost_now) else 0.0
                    tau += max(0.0, self.num(tau_now[re], 1.0)) if re < len(tau_now) else 1.0
                    cp = cp_pos.get(int(heads[re]))
                    if cp is not None and cp < len(opened):
                        factor *= max(0.0, min(1.0, self.num(opened[cp])))
                cap = min(cap, *(edge_left[re] for re in route)) * factor
                if cap <= 1e-9:
                    continue
                arcs.append([s, route, int(tails[route[0]]), int(heads[route[-1]]), cap, cost + 0.05 * tau])

            # Greedily satisfy high-penalty sinks first, using the cheapest
            # currently available source and a shortest path over usable slots.
            sink_ids = [i for i, (node, kk) in enumerate(zip(snodes, sks))
                        if kk == k and need.get((node, k), 0.0) > 1e-8]
            sink_ids.sort(key=lambda i: -max(0.0, self.num(penalties[i] if i < len(penalties) else 1.0)))
            for si in sink_ids:
                sink = snodes[si]
                remaining = need[(sink, k)]
                while remaining > 1e-8:
                    best = None
                    for origin in range(self.nn):
                        available = stock[origin, k] + supply[origin, k]
                        if origin == sink or available <= 1e-8:
                            continue
                        dist = [float("inf")] * self.nn
                        first = [-1] * self.nn
                        dist[origin] = 0.0
                        used = [False] * self.nn
                        for _ in range(self.nn):
                            u = -1
                            for v in range(self.nn):
                                if not used[v] and (u < 0 or dist[v] < dist[u]):
                                    u = v
                            if u < 0 or not np.isfinite(dist[u]):
                                break
                            used[u] = True
                            for ai, a in enumerate(arcs):
                                if a[2] != u or a[4] <= 1e-9:
                                    continue
                                v = a[3]
                                nd = dist[u] + a[5]
                                if nd < dist[v]:
                                    dist[v] = nd
                                    first[v] = ai if u == origin else first[u]
                        ai = first[sink]
                        if ai >= 0 and np.isfinite(dist[sink]):
                            if best is None or dist[sink] < best[0]:
                                best = (dist[sink], ai, origin, available)
                    if best is None:
                        break
                    _, ai, origin, available = best
                    a = arcs[ai]
                    qty = min(remaining, available, a[4], *(edge_left[re] for re in a[1]))
                    if qty <= 1e-8:
                        break
                    flows[a[0]] += qty
                    take = min(qty, stock[origin, k])
                    stock[origin, k] -= take
                    supply[origin, k] = max(0.0, supply[origin, k] - (qty - take))
                    for re in a[1]:
                        edge_left[re] = max(0.0, edge_left[re] - qty)
                    a[4] = max(0.0, a[4] - qty)
                    remaining -= qty
                    need[(sink, k)] = remaining

        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}