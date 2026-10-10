# -2.107872794253004
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
    def val(x, default=0.0):
        try:
            y = float(x)
            return y if np.isfinite(y) else default
        except Exception:
            return default

    def act(self, obs):
        flows = np.zeros(self.ns, dtype=float)
        tails, heads = self.edges["tail"], self.edges["head"]
        nedge = len(tails)
        lane_edges = self.static.get("lanes", {}).get("edges", [])
        slot_e = self.slots["edge"]
        slot_k = self.slots["k"]
        slot_lane = self.slots.get("lane", [None] * self.ns)

        stock = np.zeros((self.nn, self.nk), dtype=float)
        sq = np.asarray(obs.get("stock.qty", [])).ravel()
        for i, pair in enumerate(self.layout.get("stock_slots", [])):
            if i < len(sq):
                u, k = map(int, pair)
                stock[u, k] += max(0.0, self.val(sq[i]))
        supply = np.zeros_like(stock)
        av = np.asarray(obs.get("graph_now.supply.avail", [])).ravel()
        for i, pair in enumerate(self.layout.get("supply_slots", [])):
            if i < len(av):
                u, k = map(int, pair)
                supply[u, k] += max(0.0, self.val(av[i]))

        week = int(np.asarray(obs.get("week", [1])).ravel()[0])
        forecast = np.asarray(obs.get("demand_forecast.qty", []), dtype=float)
        backlog = np.asarray(obs.get("backlog.qty", [])).ravel()
        sinks = self.static.get("sinks", {})
        snodes = sinks.get("node", [])
        sks = sinks.get("k", [])
        penalties = sinks.get("pi", [1.0] * len(snodes))
        need = {}
        for i, (node, k) in enumerate(zip(snodes, sks)):
            node, k = int(node), int(k)
            b = max(0.0, self.val(backlog[i])) if i < len(backlog) else 0.0
            # Limit the pull horizon; subsequent weeks will refresh the plan.
            f = 0.0
            if forecast.ndim == 2 and i < forecast.shape[0]:
                f = sum(max(0.0, self.val(x)) for x in forecast[i, :min(4, forecast.shape[1])])
            need[(node, k)] = max(0.0, b + f - stock[node, k])

        # Credit shipments already scheduled to arrive at the sink within the
        # short replenishment window.
        pe = np.asarray(obs.get("pipeline.edge", [])).ravel()
        pk = np.asarray(obs.get("pipeline.k", [])).ravel()
        pq = np.asarray(obs.get("pipeline.qty", [])).ravel()
        pa = np.asarray(obs.get("pipeline.arrival_week", [])).ravel()
        live = np.asarray(obs.get("pipeline.qty.observed", np.ones(len(pq)))).ravel()
        for i in range(min(len(pe), len(pk), len(pq), len(pa))):
            if i < len(live) and not live[i]:
                continue
            e, k = int(pe[i]), int(pk[i])
            if 0 <= e < nedge and 0 <= k < self.nk and int(pa[i]) <= week + 4:
                key = (int(heads[e]), k)
                if key in need:
                    need[key] = max(0.0, need[key] - max(0.0, self.val(pq[i])))

        cap_now = np.asarray(obs.get("graph_now.u", [])).ravel()
        cost_now = np.asarray(obs.get("graph_now.c", [])).ravel()
        tau_now = np.asarray(obs.get("graph_now.tau", [])).ravel()
        tariff = np.asarray(obs.get("graph_now.tariff", []), dtype=float)
        customs = self.static.get("commodities", {}).get("v", [])
        mask = np.asarray(obs.get("action_mask", np.ones(self.ns))).ravel()
        mask_seen = bool(np.asarray(obs.get("action_mask.observed", [1])).ravel()[0])
        prohibited = np.asarray(obs.get("slot_mask", np.zeros(self.ns))).ravel()
        edge_left = np.full(nedge, np.inf)
        for e in range(nedge):
            if e < len(cap_now) and np.isfinite(cap_now[e]):
                edge_left[e] = max(0.0, float(cap_now[e]))

        for k in range(self.nk):
            arcs = []
            for s in range(self.ns):
                if int(slot_k[s]) != k:
                    continue
                allowed = (s < len(mask) and mask[s] > 0.5) if mask_seen else not (s < len(prohibited) and prohibited[s] > 0.5)
                if not allowed:
                    continue
                e = int(slot_e[s])
                route = [e]
                lane = slot_lane[s] if s < len(slot_lane) else None
                if lane is not None:
                    try:
                        le = lane_edges[int(lane)]
                        if le:
                            route = [int(x) for x in le]
                    except Exception:
                        pass
                if any(re < 0 or re >= nedge for re in route):
                    continue
                cap, weight, tau = np.inf, 0.0, 0.0
                for re in route:
                    if re < len(cap_now) and np.isfinite(cap_now[re]):
                        cap = min(cap, max(0.0, float(cap_now[re])))
                    c = self.val(cost_now[re]) if re < len(cost_now) else self.val(self.edges.get("c0", [0] * nedge)[re])
                    t = self.val(tau_now[re], 1.0) if re < len(tau_now) else 1.0
                    weight += max(0.0, c) + 0.12 * max(0.0, t)
                    tau += max(0.0, t)
                    if tariff.ndim == 2 and re < tariff.shape[0] and k < tariff.shape[1]:
                        v = self.val(customs[k]) if k < len(customs) else 0.0
                        weight += max(0.0, self.val(tariff[re, k])) * max(0.0, v)
                cap = min(cap, *(edge_left[re] for re in route))
                if cap <= 1e-9:
                    continue
                arcs.append((s, route, int(tails[route[0]]), int(heads[route[-1]]), cap, weight, tau))

            inf = 1e15
            dist = np.full((self.nn, self.nn), inf)
            first = np.full((self.nn, self.nn), -1, dtype=int)
            np.fill_diagonal(dist, 0.0)
            for ai, a in enumerate(arcs):
                _, _, u, v, cap, weight, _ = a
                if cap > 1e-9 and weight < dist[u, v]:
                    dist[u, v], first[u, v] = weight, ai
            for mid in range(self.nn):
                for u in range(self.nn):
                    if dist[u, mid] >= inf:
                        continue
                    cand = dist[u, mid] + dist[mid]
                    better = cand < dist[u] - 1e-10
                    dist[u, better] = cand[better]
                    first[u, better] = first[u, mid]

            demand_order = [i for i, (node, kk) in enumerate(zip(snodes, sks))
                            if int(kk) == k and need.get((int(node), k), 0.0) > 1e-8]
            demand_order.sort(key=lambda i: -max(0.0, self.val(penalties[i]) if i < len(penalties) else 1.0))
            for i in demand_order:
                sink = int(snodes[i])
                remaining = need[(sink, k)]
                while remaining > 1e-8:
                    choices = []
                    for origin in range(self.nn):
                        if origin == sink or dist[origin, sink] >= inf:
                            continue
                        ai = int(first[origin, sink])
                        if ai < 0:
                            continue
                        a = arcs[ai]
                        source = a[2]
                        available = stock[source, k] + supply[source, k]
                        capacity = min(a[4], *(edge_left[re] for re in a[1]))
                        if available > 1e-8 and capacity > 1e-8:
                            choices.append((dist[origin, sink], ai, available, capacity))
                    if not choices:
                        break
                    _, ai, available, capacity = min(choices)
                    s, route, source, _, _, _, _ = arcs[ai]
                    qty = min(remaining, available, capacity)
                    if qty <= 1e-8:
                        break
                    flows[s] += qty
                    from_stock = min(qty, stock[source, k])
                    stock[source, k] -= from_stock
                    supply[source, k] = max(0.0, supply[source, k] - (qty - from_stock))
                    for re in route:
                        edge_left[re] = max(0.0, edge_left[re] - qty)
                    a = arcs[ai]
                    arcs[ai] = (a[0], a[1], a[2], a[3], max(0.0, a[4] - qty), a[5], a[6])
                    remaining -= qty
                    need[(sink, k)] = remaining

        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}