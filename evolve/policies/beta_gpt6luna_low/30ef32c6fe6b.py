# -2.108031991680676
import heapq
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
            v = float(x)
            return v if np.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    def act(self, obs):
        flows = np.zeros(self.ns, dtype=float)
        tails = self.edges["tail"]
        heads = self.edges["head"]
        ne = len(tails)
        lane_edges = self.static.get("lanes", {}).get("edges", [])

        stock = np.zeros((self.nn, self.nk), dtype=float)
        sq = np.asarray(obs.get("stock.qty", [])).ravel()
        for i, pair in enumerate(self.layout.get("stock_slots", [])):
            if i < len(sq):
                u, k = map(int, pair)
                stock[u, k] += max(0.0, self.num(sq[i]))
        supply = np.zeros_like(stock)
        av = np.asarray(obs.get("graph_now.supply.avail", [])).ravel()
        for i, pair in enumerate(self.layout.get("supply_slots", [])):
            if i < len(av):
                u, k = map(int, pair)
                supply[u, k] += max(0.0, self.num(av[i]))

        sinks = self.static.get("sinks", {})
        snodes = [int(x) for x in sinks.get("node", [])]
        sks = [int(x) for x in sinks.get("k", [])]
        penalties = sinks.get("pi", [1.0] * len(snodes))
        backlog = np.asarray(obs.get("backlog.qty", [])).ravel()
        forecast = np.asarray(obs.get("demand_forecast.qty", []), dtype=float)
        need = {}
        for i, (u, k) in enumerate(zip(snodes, sks)):
            h = min(3, forecast.shape[1]) if forecast.ndim == 2 and i < forecast.shape[0] else 0
            f = sum(max(0.0, self.num(x)) for x in forecast[i, :h]) if h else 0.0
            b = max(0.0, self.num(backlog[i])) if i < len(backlog) else 0.0
            need[(u, k)] = max(0.0, b + f - stock[u, k])

        week_arr = np.asarray(obs.get("week", [1])).ravel()
        week = int(week_arr[0]) if len(week_arr) else 1
        pe = np.asarray(obs.get("pipeline.edge", [])).ravel()
        pk = np.asarray(obs.get("pipeline.k", [])).ravel()
        pq = np.asarray(obs.get("pipeline.qty", [])).ravel()
        pa = np.asarray(obs.get("pipeline.arrival_week", [])).ravel()
        live = np.asarray(obs.get("pipeline.qty.observed", np.ones(len(pq)))).ravel()
        for i in range(min(len(pe), len(pk), len(pq), len(pa))):
            if i < len(live) and not live[i]:
                continue
            e, k = int(pe[i]), int(pk[i])
            if 0 <= e < ne and 0 <= k < self.nk and int(pa[i]) <= week + 3:
                key = (int(heads[e]), k)
                if key in need:
                    need[key] = max(0.0, need[key] - max(0.0, self.num(pq[i])))

        caps = np.asarray(obs.get("graph_now.u", [])).ravel()
        costs = np.asarray(obs.get("graph_now.c", [])).ravel()
        taus = np.asarray(obs.get("graph_now.tau", [])).ravel()
        tariffs = np.asarray(obs.get("graph_now.tariff", []), dtype=float)
        values = self.static.get("commodities", {}).get("v", [])
        mask = np.asarray(obs.get("action_mask", np.ones(self.ns))).ravel()
        mask_seen = bool(np.asarray(obs.get("action_mask.observed", [1])).ravel()[0])
        prohibited = np.asarray(obs.get("slot_mask", np.zeros(self.ns))).ravel()
        slot_e = self.slots["edge"]
        slot_k = self.slots["k"]
        slot_lane = self.slots.get("lane", [None] * self.ns)
        edge_left = np.full(ne, np.inf)
        for e in range(min(ne, len(caps))):
            if np.isfinite(caps[e]):
                edge_left[e] = max(0.0, float(caps[e]))

        for k in range(self.nk):
            arcs = []
            adj = [[] for _ in range(self.nn)]
            for s in range(self.ns):
                if int(slot_k[s]) != k:
                    continue
                allowed = (s < len(mask) and mask[s] > 0.5) if mask_seen else not (s < len(prohibited) and prohibited[s] > 0.5)
                if not allowed:
                    continue
                route = [int(slot_e[s])]
                lane = slot_lane[s] if s < len(slot_lane) else None
                if lane is not None:
                    try:
                        le = lane_edges[int(lane)]
                        if le:
                            route = [int(e) for e in le]
                    except (TypeError, ValueError, IndexError):
                        pass
                if not route or any(e < 0 or e >= ne for e in route):
                    continue
                cap = min(edge_left[e] for e in route)
                cost = tau = 0.0
                for e in route:
                    if e < len(caps) and np.isfinite(caps[e]):
                        cap = min(cap, max(0.0, float(caps[e])))
                    if e < len(costs):
                        cost += max(0.0, self.num(costs[e]))
                    if e < len(taus):
                        tau += max(0.0, self.num(taus[e], 1.0))
                    if tariffs.ndim == 2 and e < tariffs.shape[0] and k < tariffs.shape[1]:
                        val = self.num(values[k]) if k < len(values) else 0.0
                        cost += max(0.0, self.num(tariffs[e, k])) * max(0.0, val)
                if cap <= 1e-9:
                    continue
                u, v = int(tails[route[0]]), int(heads[route[-1]])
                ai = len(arcs)
                arcs.append([s, route, u, v, cap, cost + 0.15 * tau])
                adj[u].append(ai)

            order = [i for i, kk in enumerate(sks) if kk == k and need.get((snodes[i], k), 0.0) > 1e-8]
            order.sort(key=lambda i: -max(0.0, self.num(penalties[i] if i < len(penalties) else 1.0)))
            for si in order:
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
                        heap = [(0.0, origin)]
                        while heap:
                            du, u = heapq.heappop(heap)
                            if du > dist[u] + 1e-10:
                                continue
                            for ai in adj[u]:
                                a = arcs[ai]
                                if a[4] <= 1e-9:
                                    continue
                                nd = du + a[5]
                                if nd < dist[a[3]] - 1e-10:
                                    dist[a[3]] = nd
                                    first[a[3]] = ai if u == origin else first[u]
                                    heapq.heappush(heap, (nd, a[3]))
                        ai = first[sink]
                        if ai < 0 or not np.isfinite(dist[sink]):
                            continue
                        a = arcs[ai]
                        src = a[2]
                        avail_src = stock[src, k] + supply[src, k]
                        if avail_src <= 1e-8:
                            continue
                        candidate = (dist[sink], origin, ai, avail_src)
                        if best is None or candidate[:2] < best[:2]:
                            best = candidate
                    if best is None:
                        break
                    _, _, ai, available = best
                    a = arcs[ai]
                    s, route, src, _, arc_cap, _ = a
                    qty = min(remaining, available, arc_cap, *(edge_left[e] for e in route))
                    if qty <= 1e-8:
                        break
                    flows[s] += qty
                    take = min(qty, stock[src, k])
                    stock[src, k] -= take
                    supply[src, k] = max(0.0, supply[src, k] - (qty - take))
                    for e in route:
                        edge_left[e] = max(0.0, edge_left[e] - qty)
                    a[4] = max(0.0, a[4] - qty)
                    remaining -= qty
                need[(sink, k)] = remaining

        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}