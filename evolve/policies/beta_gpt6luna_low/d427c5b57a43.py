# -2.108697009571195
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
        self.nominal = self._read_nominal(config)

    @staticmethod
    def val(x, default=0.0):
        try:
            x = float(x)
            return x if np.isfinite(x) else default
        except Exception:
            return default

    def _read_nominal(self, config):
        # Routing table encodings vary by instance; accept only recognizable
        # lists of [node, commodity, destination] or [node, commodity, edge].
        out = {}
        table = self.static.get("instance", {}).get("routing_table", [])
        if isinstance(table, list):
            for row in table:
                if isinstance(row, dict):
                    try:
                        n = int(row.get("node", row.get("from")))
                        k = int(row.get("k", row.get("commodity")))
                        d = row.get("destination", row.get("to"))
                        if d is not None:
                            out.setdefault((n, k), set()).add(int(d))
                    except Exception:
                        pass
                elif isinstance(row, (list, tuple)) and len(row) >= 3:
                    try:
                        out.setdefault((int(row[0]), int(row[1])), set()).add(int(row[2]))
                    except Exception:
                        pass
        return out

    def act(self, obs):
        n, nk, ns = self.nn, self.nk, self.ns
        flows = np.zeros(ns, dtype=float)
        tails, heads = self.edges["tail"], self.edges["head"]
        slot_e, slot_k = self.slots["edge"], self.slots["k"]
        slot_l = self.slots.get("lane", [None] * ns)

        stock = np.zeros((n, nk), dtype=float)
        raw = np.asarray(obs.get("stock.qty", [])).ravel()
        for i, pair in enumerate(self.layout.get("stock_slots", [])):
            if i < len(raw):
                u, k = map(int, pair)
                stock[u, k] += max(0.0, self.val(raw[i]))
        supply = np.zeros_like(stock)
        raw = np.asarray(obs.get("graph_now.supply.avail", [])).ravel()
        for i, pair in enumerate(self.layout.get("supply_slots", [])):
            if i < len(raw):
                u, k = map(int, pair)
                supply[u, k] += max(0.0, self.val(raw[i]))

        sinks = self.static.get("sinks", {})
        sn = [int(x) for x in sinks.get("node", [])]
        sk = [int(x) for x in sinks.get("k", [])]
        forecast = np.asarray(obs.get("demand_forecast.qty", []), dtype=float)
        backlog = np.asarray(obs.get("backlog.qty", [])).ravel()
        week = int(np.asarray(obs.get("week", [1])).ravel()[0])
        target = {}
        for i, (u, k) in enumerate(zip(sn, sk)):
            f = 0.0
            if forecast.ndim == 2 and i < forecast.shape[0]:
                h = min(2, forecast.shape[1])
                f = sum(max(0.0, self.val(x)) for x in forecast[i, :h])
            b = max(0.0, self.val(backlog[i])) if i < len(backlog) else 0.0
            target[(u, k)] = b + f

        # Account for shipments already expected to arrive at sinks soon.
        pe = np.asarray(obs.get("pipeline.edge", [])).ravel()
        pk = np.asarray(obs.get("pipeline.k", [])).ravel()
        pq = np.asarray(obs.get("pipeline.qty", [])).ravel()
        pa = np.asarray(obs.get("pipeline.arrival_week", [])).ravel()
        live = np.asarray(obs.get("pipeline.qty.observed", np.ones(len(pq)))).ravel()
        for i in range(min(len(pe), len(pk), len(pq), len(pa))):
            if i < len(live) and not live[i]:
                continue
            e, k = int(pe[i]), int(pk[i])
            if 0 <= e < self.ne and 0 <= k < nk and int(pa[i]) <= week + 2:
                key = (int(heads[e]), k)
                if key in target:
                    target[key] = max(0.0, target[key] - max(0.0, self.val(pq[i])))
        for key in list(target):
            u, k = key
            target[key] = max(0.0, target[key] - stock[u, k])

        caps = np.asarray(obs.get("graph_now.u", [])).ravel()
        costs = np.asarray(obs.get("graph_now.c", [])).ravel()
        taus = np.asarray(obs.get("graph_now.tau", [])).ravel()
        tariffs = np.asarray(obs.get("graph_now.tariff", []), dtype=float)
        vals = self.static.get("commodities", {}).get("v", [])
        mask = np.asarray(obs.get("action_mask", np.ones(ns))).ravel()
        mask_seen = bool(np.asarray(obs.get("action_mask.observed", [1])).ravel()[0])
        banned = np.asarray(obs.get("slot_mask", np.zeros(ns))).ravel()
        edge_left = np.full(self.ne, np.inf)
        for e in range(min(self.ne, len(caps))):
            if np.isfinite(caps[e]):
                edge_left[e] = max(0.0, float(caps[e]))

        for k in range(nk):
            arcs = []
            adj = [[] for _ in range(n)]
            for s in range(ns):
                if int(slot_k[s]) != k:
                    continue
                allowed = mask[s] > 0.5 if mask_seen and s < len(mask) else not (s < len(banned) and banned[s] > 0)
                if not allowed:
                    continue
                route = [int(slot_e[s])]
                lane = slot_l[s] if s < len(slot_l) else None
                if lane is not None:
                    try:
                        if self.lane_edges[int(lane)]:
                            route = [int(e) for e in self.lane_edges[int(lane)]]
                    except Exception:
                        pass
                if not route or any(e < 0 or e >= self.ne for e in route):
                    continue
                cap = min(edge_left[e] for e in route)
                weight = 0.0
                for e in route:
                    c = self.val(costs[e]) if e < len(costs) else 0.0
                    t = self.val(taus[e], 1.0) if e < len(taus) else 1.0
                    weight += max(0.0, c) + 0.08 * max(0.0, t)
                    if tariffs.ndim == 2 and e < tariffs.shape[0] and k < tariffs.shape[1]:
                        v = self.val(vals[k]) if k < len(vals) else 0.0
                        weight += max(0.0, self.val(tariffs[e, k])) * max(0.0, v)
                if cap <= 1e-9:
                    continue
                u, v = int(tails[route[0]]), int(heads[route[-1]])
                ai = len(arcs)
                arcs.append([s, route, u, v, cap, weight])
                adj[u].append(ai)

            # For each sink need, find a currently stocked/supplied origin and
            # dispatch one first-hop route. Recompute after each allocation.
            demands = [(u, target[(u, k)]) for u, kk in zip(sn, sk)
                       if kk == k and target.get((u, k), 0.0) > 1e-8]
            demands.sort(key=lambda x: -x[1])
            for sink, remaining0 in demands:
                remaining = remaining0
                while remaining > 1e-8:
                    best = None
                    for origin in range(n):
                        available = stock[origin, k] + supply[origin, k]
                        if origin == sink or available <= 1e-8:
                            continue
                        dist = [float("inf")] * n
                        first = [-1] * n
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
                                if nd < dist[a[3]]:
                                    dist[a[3]] = nd
                                    first[a[3]] = ai if u == origin else first[u]
                                    heapq.heappush(heap, (nd, a[3]))
                        ai = first[sink]
                        if ai < 0 or not np.isfinite(dist[sink]):
                            continue
                        cand = (dist[sink], origin, ai, available)
                        if best is None or cand[:2] < best[:2]:
                            best = cand
                    if best is None:
                        break
                    _, origin, ai, available = best
                    a = arcs[ai]
                    qty = min(remaining, available, a[4], *(edge_left[e] for e in a[1]))
                    if qty <= 1e-8:
                        break
                    flows[a[0]] += qty
                    use = min(qty, stock[origin, k])
                    stock[origin, k] -= use
                    supply[origin, k] = max(0.0, supply[origin, k] - (qty - use))
                    for e in a[1]:
                        edge_left[e] = max(0.0, edge_left[e] - qty)
                    a[4] = max(0.0, a[4] - qty)
                    remaining -= qty
        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}