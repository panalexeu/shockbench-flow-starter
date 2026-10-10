# -2.1087656307835307
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
            y = float(x)
            return y if np.isfinite(y) else default
        except (TypeError, ValueError):
            return default

    def act(self, obs):
        flows = np.zeros(self.ns, dtype=float)
        tails, heads = self.edges["tail"], self.edges["head"]
        nedge = len(tails)
        lane_edges = self.static.get("lanes", {}).get("edges", [])
        slot_edge = self.slots["edge"]
        slot_k = self.slots["k"]
        slot_lane = self.slots.get("lane", [None] * self.ns)

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
        need = {}
        for i, (node, k) in enumerate(zip(snodes, sks)):
            f = 0.0
            if forecast.ndim == 2 and i < forecast.shape[0]:
                f = sum(max(0.0, self.num(x)) for x in forecast[i, :min(2, forecast.shape[1])])
            b = max(0.0, self.num(backlog[i])) if i < len(backlog) else 0.0
            need[(node, k)] = max(0.0, b + f - stock[node, k])

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
            if 0 <= e < nedge and 0 <= k < self.nk and int(pa[i]) <= week + 2:
                key = (int(heads[e]), k)
                if key in need:
                    need[key] = max(0.0, need[key] - max(0.0, self.num(pq[i])))

        cap = np.asarray(obs.get("graph_now.u", [])).ravel()
        cost = np.asarray(obs.get("graph_now.c", [])).ravel()
        tau = np.asarray(obs.get("graph_now.tau", [])).ravel()
        tariff = np.asarray(obs.get("graph_now.tariff", []), dtype=float)
        values = self.static.get("commodities", {}).get("v", [])
        opened = np.asarray(obs.get("graph_now.open", [])).ravel()
        cp_pos = {int(node): i for i, node in enumerate(self.layout.get("chokepoints", []))}
        mask = np.asarray(obs.get("action_mask", np.ones(self.ns))).ravel()
        mask_obs = np.asarray(obs.get("action_mask.observed", [1])).ravel()
        prohibited = np.asarray(obs.get("slot_mask", np.zeros(self.ns))).ravel()
        mask_seen = bool(mask_obs[0]) if len(mask_obs) else True

        edge_left = np.full(nedge, np.inf)
        for e in range(min(nedge, len(cap))):
            if np.isfinite(cap[e]):
                edge_left[e] = max(0.0, float(cap[e]))

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
                        r = lane_edges[int(lane)]
                        if r:
                            route = [int(e) for e in r]
                    except (TypeError, ValueError, IndexError):
                        pass
                if not route or any(e < 0 or e >= nedge for e in route):
                    continue
                u, v = int(tails[route[0]]), int(heads[route[-1]])
                route_cap = min(edge_left[e] for e in route)
                weight = 0.0
                for e in route:
                    if e < len(cap) and np.isfinite(cap[e]):
                        route_cap = min(route_cap, max(0.0, float(cap[e])))
                    if e < len(cost):
                        weight += max(0.0, self.num(cost[e]))
                    if e < len(tau):
                        weight += 0.2 * max(0.0, self.num(tau[e]))
                    cp = cp_pos.get(int(heads[e]))
                    if cp is not None and cp < len(opened):
                        route_cap *= max(0.0, min(1.0, self.num(opened[cp])))
                    if tariff.ndim == 2 and e < tariff.shape[0] and k < tariff.shape[1]:
                        val = self.num(values[k]) if k < len(values) else 0.0
                        weight += max(0.0, self.num(tariff[e, k])) * max(0.0, val)
                if route_cap <= 1e-9:
                    continue
                ai = len(arcs)
                arcs.append([s, route, u, v, route_cap, weight])
                adj[u].append((v, ai))

            # Prioritize high-penalty sinks; repeatedly send from the cheapest
            # stocked or supplied origin along the currently cheapest path.
            order = [i for i, kk in enumerate(sks) if kk == k and need.get((snodes[i], k), 0.0) > 1e-8]
            order.sort(key=lambda i: -max(0.0, self.num(penalties[i]) if i < len(penalties) else 1.0))
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
                        if ai >= 0 and np.isfinite(dist[sink]):
                            candidate = (dist[sink], origin, ai, available)
                            if best is None or candidate[:2] < best[:2]:
                                best = candidate
                    if best is None:
                        break
                    _, origin, ai, available = best
                    a = arcs[ai]
                    qty = min(remaining, available, a[4], *(edge_left[e] for e in a[1]))
                    if qty <= 1e-8:
                        break
                    flows[a[0]] += qty
                    take = min(qty, stock[origin, k])
                    stock[origin, k] -= take
                    supply[origin, k] = max(0.0, supply[origin, k] - (qty - take))
                    for e in a[1]:
                        edge_left[e] = max(0.0, edge_left[e] - qty)
                    a[4] = max(0.0, a[4] - qty)
                    remaining -= qty
                need[(sink, k)] = remaining

        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}
