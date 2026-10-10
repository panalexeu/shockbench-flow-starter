# -2.107877857875421
import heapq
import numpy as np


class Agent:
    def __init__(self, config=None):
        self.static = config["static"]
        self.layout = config["layout"]
        self.edges = self.static["edges"]
        self.slots = self.static["action_slots"]
        self.nslot = len(self.slots["edge"])
        self.nn = len(self.static["nodes"]["id"])
        self.nk = len(self.static["commodities"]["id"])
        action = config["spaces"]["action"]
        self.override_qty = np.zeros(action["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action["release_mode"]["shape"], dtype=np.int64)

    @staticmethod
    def num(x, default=0.0):
        try:
            y = float(x)
            return y if np.isfinite(y) else default
        except (TypeError, ValueError):
            return default

    def act(self, obs):
        ns, nn, nk = self.nslot, self.nn, self.nk
        flows = np.zeros(ns, dtype=float)
        tails, heads = self.edges["tail"], self.edges["head"]
        lane_edges = self.static.get("lanes", {}).get("edges", [])
        slot_edge = self.slots["edge"]
        slot_k = self.slots["k"]
        slot_lane = self.slots.get("lane", [None] * ns)

        stock = np.zeros((nn, nk), dtype=float)
        q = np.asarray(obs.get("stock.qty", [])).ravel()
        for i, pair in enumerate(self.layout.get("stock_slots", [])):
            if i < len(q):
                n, k = map(int, pair)
                stock[n, k] += max(0.0, self.num(q[i]))
        supply = np.zeros_like(stock)
        q = np.asarray(obs.get("graph_now.supply.avail", [])).ravel()
        for i, pair in enumerate(self.layout.get("supply_slots", [])):
            if i < len(q):
                n, k = map(int, pair)
                supply[n, k] += max(0.0, self.num(q[i]))

        sink_table = self.static.get("sinks", {})
        sink_nodes = [int(x) for x in sink_table.get("node", [])]
        sink_ks = [int(x) for x in sink_table.get("k", [])]
        penalties = sink_table.get("pi", [1.0] * len(sink_nodes))
        forecast = np.asarray(obs.get("demand_forecast.qty", []), dtype=float)
        backlog = np.asarray(obs.get("backlog.qty", [])).ravel()
        week = int(np.asarray(obs.get("week", [1])).ravel()[0])
        # Cover a useful replenishment window, but do not treat the full forecast
        # horizon as an order to ship immediately.
        need = {}
        for i, (n, k) in enumerate(zip(sink_nodes, sink_ks)):
            f = 0.0
            if forecast.ndim == 2 and i < forecast.shape[0]:
                f = sum(max(0.0, self.num(x)) for x in forecast[i, :min(4, forecast.shape[1])])
            b = max(0.0, self.num(backlog[i])) if i < len(backlog) else 0.0
            need[(n, k)] = b + f

        # Credit only cargo whose current edge delivers directly to the sink.
        pe = np.asarray(obs.get("pipeline.edge", [])).ravel()
        pk = np.asarray(obs.get("pipeline.k", [])).ravel()
        pq = np.asarray(obs.get("pipeline.qty", [])).ravel()
        pa = np.asarray(obs.get("pipeline.arrival_week", [])).ravel()
        live = np.asarray(obs.get("pipeline.qty.observed", np.ones(len(pq)))).ravel()
        for i in range(min(len(pe), len(pk), len(pq), len(pa))):
            if i < len(live) and not live[i]:
                continue
            e, k = int(pe[i]), int(pk[i])
            if 0 <= e < len(heads) and 0 <= k < nk and int(pa[i]) <= week + 4:
                key = (int(heads[e]), k)
                if key in need:
                    need[key] = max(0.0, need[key] - max(0.0, self.num(pq[i])))
        for i, (n, k) in enumerate(zip(sink_nodes, sink_ks)):
            need[(n, k)] = max(0.0, need[(n, k)] - stock[n, k])

        cap_now = np.asarray(obs.get("graph_now.u", [])).ravel()
        cost_now = np.asarray(obs.get("graph_now.c", [])).ravel()
        tau_now = np.asarray(obs.get("graph_now.tau", [])).ravel()
        tariff = np.asarray(obs.get("graph_now.tariff", []), dtype=float)
        values = self.static.get("commodities", {}).get("v", [])
        mask = np.asarray(obs.get("action_mask", np.ones(ns))).ravel()
        mask_seen_arr = np.asarray(obs.get("action_mask.observed", [1])).ravel()
        mask_seen = bool(mask_seen_arr[0]) if len(mask_seen_arr) else True
        prohibited = np.asarray(obs.get("slot_mask", np.zeros(ns))).ravel()
        edge_left = np.full(len(tails), np.inf)
        for e in range(min(len(tails), len(cap_now))):
            if np.isfinite(cap_now[e]):
                edge_left[e] = max(0.0, float(cap_now[e]))

        for k in range(nk):
            arcs = []
            adj = [[] for _ in range(nn)]
            for s in range(ns):
                if int(slot_k[s]) != k:
                    continue
                allowed = (s < len(mask) and mask[s] > 0.5) if mask_seen else not (s < len(prohibited) and prohibited[s] > 0.5)
                if not allowed:
                    continue
                route = [int(slot_edge[s])]
                lane = slot_lane[s] if s < len(slot_lane) else None
                if lane is not None:
                    try:
                        le = lane_edges[int(lane)]
                        if le:
                            route = [int(e) for e in le]
                    except (TypeError, ValueError, IndexError):
                        pass
                if not route or any(e < 0 or e >= len(tails) for e in route):
                    continue
                cap, weight = float("inf"), 0.0
                for e in route:
                    cap = min(cap, edge_left[e])
                    c = self.num(cost_now[e]) if e < len(cost_now) else self.num(self.edges.get("c0", [0] * len(tails))[e])
                    tau = self.num(tau_now[e], 1.0) if e < len(tau_now) else 1.0
                    weight += max(0.0, c) + 0.1 * max(0.0, tau)
                    if tariff.ndim == 2 and e < tariff.shape[0] and k < tariff.shape[1]:
                        value = self.num(values[k]) if k < len(values) else 0.0
                        weight += max(0.0, self.num(tariff[e, k])) * max(0.0, value)
                if cap <= 1e-9:
                    continue
                u, v = int(tails[route[0]]), int(heads[route[-1]])
                ai = len(arcs)
                arcs.append([s, route, u, v, cap, weight])
                adj[u].append((v, ai))

            sinks = []
            for i, (n, kk) in enumerate(zip(sink_nodes, sink_ks)):
                d = need.get((n, kk), 0.0)
                if kk == k and d > 1e-8:
                    pi = max(0.0, self.num(penalties[i] if i < len(penalties) else 1.0))
                    sinks.append((n, d, pi))
            sinks.sort(key=lambda x: (-x[2], -x[1]))

            for sink, demand, _ in sinks:
                remaining = demand
                while remaining > 1e-8:
                    best = None
                    for origin in range(nn):
                        available = stock[origin, k] + supply[origin, k]
                        if origin == sink or available <= 1e-8:
                            continue
                        dist = [float("inf")] * nn
                        first = [-1] * nn
                        dist[origin] = 0.0
                        heap = [(0.0, origin)]
                        while heap:
                            du, u = heapq.heappop(heap)
                            if du > dist[u] + 1e-10:
                                continue
                            for v, ai in adj[u]:
                                a = arcs[ai]
                                if a[4] <= 1e-9:
                                    continue
                                nd = du + a[5]
                                if nd < dist[v] - 1e-10:
                                    dist[v] = nd
                                    first[v] = ai if u == origin else first[u]
                                    heapq.heappush(heap, (nd, v))
                        ai = first[sink]
                        if ai < 0 or not np.isfinite(dist[sink]):
                            continue
                        a = arcs[ai]
                        qty_cap = min(a[4], *(edge_left[e] for e in a[1]))
                        if qty_cap <= 1e-9:
                            continue
                        cand = (dist[sink], origin, ai, available, qty_cap)
                        if best is None or cand[:2] < best[:2]:
                            best = cand
                    if best is None:
                        break
                    _, origin, ai, available, qty_cap = best
                    a = arcs[ai]
                    qty = min(remaining, available, qty_cap)
                    if qty <= 1e-9:
                        break
                    flows[a[0]] += qty
                    take = min(qty, stock[origin, k])
                    stock[origin, k] -= take
                    supply[origin, k] = max(0.0, supply[origin, k] - (qty - take))
                    for e in a[1]:
                        edge_left[e] = max(0.0, edge_left[e] - qty)
                    a[4] = max(0.0, a[4] - qty)
                    remaining -= qty

        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}
