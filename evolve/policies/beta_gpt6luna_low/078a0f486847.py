# -2.1080279433737554
import heapq
import numpy as np


class Agent:
    def __init__(self, config=None):
        self.static = config["static"]
        self.layout = config["layout"]
        self.edges = self.static["edges"]
        self.slots = self.static["action_slots"]
        self.lane_edges = self.static.get("lanes", {}).get("edges", [])
        self.nn = len(self.static["nodes"]["id"])
        self.nk = len(self.static["commodities"]["id"])
        self.ne = len(self.edges["tail"])
        self.ns = len(self.slots["edge"])
        action = config["spaces"]["action"]
        self.override_qty = np.zeros(action["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action["release_mode"]["shape"], dtype=np.int64)

    @staticmethod
    def val(x, default=0.0):
        try:
            y = float(x)
            return y if np.isfinite(y) else default
        except (TypeError, ValueError):
            return default

    def act(self, obs):
        flows = np.zeros(self.ns, dtype=float)
        tails = self.edges["tail"]
        heads = self.edges["head"]
        stock = np.zeros((self.nn, self.nk), dtype=float)
        supply = np.zeros_like(stock)

        raw = np.asarray(obs.get("stock.qty", [])).ravel()
        for i, pair in enumerate(self.layout.get("stock_slots", [])):
            if i < len(raw):
                n, k = map(int, pair)
                stock[n, k] += max(0.0, self.val(raw[i]))
        raw = np.asarray(obs.get("graph_now.supply.avail", [])).ravel()
        for i, pair in enumerate(self.layout.get("supply_slots", [])):
            if i < len(raw):
                n, k = map(int, pair)
                supply[n, k] += max(0.0, self.val(raw[i]))

        week_arr = np.asarray(obs.get("week", [1])).ravel()
        week = int(week_arr[0]) if len(week_arr) else 1
        sinks = self.static.get("sinks", {})
        sink_nodes = [int(x) for x in sinks.get("node", [])]
        sink_ks = [int(x) for x in sinks.get("k", [])]
        penalties = sinks.get("pi", [1.0] * len(sink_nodes))
        backlog = np.asarray(obs.get("backlog.qty", [])).ravel()
        forecast = np.asarray(obs.get("demand_forecast.qty", []), dtype=float)
        need = {}
        for i, (n, k) in enumerate(zip(sink_nodes, sink_ks)):
            b = max(0.0, self.val(backlog[i])) if i < len(backlog) else 0.0
            # Cover current and near-term demand; only dispatch when there is a
            # real shortfall, rather than trying to build a large stockpile.
            h = min(3, forecast.shape[1]) if forecast.ndim == 2 and i < forecast.shape[0] else 0
            f = sum(max(0.0, self.val(x)) for x in forecast[i, :h]) if h else 0.0
            need[(n, k)] = max(0.0, b + f - stock[n, k])

        # Subtract shipments that are scheduled to reach their final sink soon.
        pe = np.asarray(obs.get("pipeline.edge", [])).ravel()
        pk = np.asarray(obs.get("pipeline.k", [])).ravel()
        pq = np.asarray(obs.get("pipeline.qty", [])).ravel()
        pa = np.asarray(obs.get("pipeline.arrival_week", [])).ravel()
        live = np.asarray(obs.get("pipeline.qty.observed", np.ones(len(pq)))).ravel()
        sink_keys = set(need)
        for i in range(min(len(pe), len(pk), len(pq), len(pa))):
            if i < len(live) and not live[i]:
                continue
            e, k = int(pe[i]), int(pk[i])
            if 0 <= e < self.ne and 0 <= k < self.nk and int(pa[i]) <= week + 3:
                key = (int(heads[e]), k)
                if key in sink_keys:
                    need[key] = max(0.0, need[key] - max(0.0, self.val(pq[i])))

        cap_now = np.asarray(obs.get("graph_now.u", [])).ravel()
        cost_now = np.asarray(obs.get("graph_now.c", [])).ravel()
        tau_now = np.asarray(obs.get("graph_now.tau", [])).ravel()
        mask = np.asarray(obs.get("action_mask", np.ones(self.ns))).ravel()
        mask_seen = bool(np.asarray(obs.get("action_mask.observed", [1])).ravel()[0])
        prohibited = np.asarray(obs.get("slot_mask", np.zeros(self.ns))).ravel()
        edge_left = np.full(self.ne, np.inf)
        for e in range(min(self.ne, len(cap_now))):
            if np.isfinite(cap_now[e]):
                edge_left[e] = max(0.0, float(cap_now[e]))

        slot_edge = self.slots["edge"]
        slot_k = self.slots["k"]
        slot_lane = self.slots.get("lane", [None] * self.ns)

        for k in range(self.nk):
            arcs = []
            adj = [[] for _ in range(self.nn)]
            for s in range(self.ns):
                if int(slot_k[s]) != k:
                    continue
                allowed = mask[s] > 0.5 if mask_seen and s < len(mask) else not (s < len(prohibited) and prohibited[s] > 0.5)
                if not allowed:
                    continue
                route = [int(slot_edge[s])]
                lane = slot_lane[s] if s < len(slot_lane) else None
                if lane is not None:
                    try:
                        candidate = self.lane_edges[int(lane)]
                        if candidate:
                            route = [int(e) for e in candidate]
                    except (TypeError, ValueError, IndexError):
                        pass
                if not route or any(e < 0 or e >= self.ne for e in route):
                    continue
                cap = min(edge_left[e] for e in route)
                cost = 0.0
                tau = 0.0
                for e in route:
                    if e < len(cap_now) and np.isfinite(cap_now[e]):
                        cap = min(cap, max(0.0, float(cap_now[e])))
                    if e < len(cost_now):
                        cost += max(0.0, self.val(cost_now[e]))
                    if e < len(tau_now):
                        tau += max(0.0, self.val(tau_now[e], 1.0))
                if cap <= 1e-9:
                    continue
                u, v = int(tails[route[0]]), int(heads[route[-1]])
                ai = len(arcs)
                # A small delay penalty avoids needlessly slow alternatives.
                arcs.append([s, route, u, v, cap, cost + 0.15 * tau])
                adj[u].append(ai)

            # Process more costly-to-miss sinks first.
            order = [i for i, kk in enumerate(sink_ks)
                     if kk == k and need.get((sink_nodes[i], k), 0.0) > 1e-8]
            order.sort(key=lambda i: -max(0.0, self.val(penalties[i]) if i < len(penalties) else 1.0))
            for si in order:
                sink = sink_nodes[si]
                remaining = need.get((sink, k), 0.0)
                while remaining > 1e-8:
                    best = None
                    # Find the least-cost reachable source with stock or supply.
                    for origin in range(self.nn):
                        available = stock[origin, k] + supply[origin, k]
                        if origin == sink or available <= 1e-8:
                            continue
                        dist = [float("inf")] * self.nn
                        first = [-1] * self.nn
                        dist[origin] = 0.0
                        heap = [(0.0, origin)]
                        while heap:
                            d, u = heapq.heappop(heap)
                            if d > dist[u] + 1e-10:
                                continue
                            for ai in adj[u]:
                                a = arcs[ai]
                                if a[4] <= 1e-9:
                                    continue
                                nd = d + a[5]
                                if nd < dist[a[3]] - 1e-10:
                                    dist[a[3]] = nd
                                    first[a[3]] = ai if u == origin else first[u]
                                    heapq.heappush(heap, (nd, a[3]))
                        ai = first[sink]
                        if ai < 0 or not np.isfinite(dist[sink]):
                            continue
                        a = arcs[ai]
                        capacity = min(a[4], *(edge_left[e] for e in a[1]))
                        candidate = (dist[sink], origin, ai, available, capacity)
                        if capacity > 1e-9 and (best is None or candidate[:2] < best[:2]):
                            best = candidate
                    if best is None:
                        break
                    _, origin, ai, available, capacity = best
                    a = arcs[ai]
                    qty = min(remaining, available, capacity)
                    if qty <= 1e-9:
                        break
                    flows[a[0]] += qty
                    used = min(qty, stock[origin, k])
                    stock[origin, k] -= used
                    supply[origin, k] = max(0.0, supply[origin, k] - (qty - used))
                    for e in a[1]:
                        edge_left[e] = max(0.0, edge_left[e] - qty)
                    a[4] = max(0.0, a[4] - qty)
                    remaining -= qty
                need[(sink, k)] = remaining

        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}
