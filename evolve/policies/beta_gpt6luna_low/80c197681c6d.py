# -2.108102180449284
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
        spaces = config["spaces"]["action"]
        self.override_qty = np.zeros(spaces["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(spaces["release_mode"]["shape"], dtype=np.int64)

    @staticmethod
    def num(x, default=0.0):
        try:
            y = float(x)
            return y if np.isfinite(y) else default
        except Exception:
            return default

    def act(self, obs):
        flows = np.zeros(self.ns, dtype=float)
        tails, heads = self.edges["tail"], self.edges["head"]
        lane_edges = self.static.get("lanes", {}).get("edges", [])
        slot_e = self.slots["edge"]
        slot_k = self.slots["k"]
        slot_lane = self.slots.get("lane", [None] * self.ns)

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

        week = int(np.asarray(obs.get("week", [1])).ravel()[0])
        sinktab = self.static.get("sinks", {})
        snodes = [int(x) for x in sinktab.get("node", [])]
        sks = [int(x) for x in sinktab.get("k", [])]
        penalties = sinktab.get("pi", [1.0] * len(snodes))
        backlog = np.asarray(obs.get("backlog.qty", [])).ravel()
        forecast = np.asarray(obs.get("demand_forecast.qty", []), dtype=float)
        # Plan a short replenishment window; do not fill sinks with the full forecast horizon.
        need = {}
        for i, (u, k) in enumerate(zip(snodes, sks)):
            f = 0.0
            if forecast.ndim == 2 and i < forecast.shape[0]:
                f = sum(max(0.0, self.num(x)) for x in forecast[i, :min(3, forecast.shape[1])])
            b = max(0.0, self.num(backlog[i])) if i < len(backlog) else 0.0
            need[(u, k)] = max(0.0, f + b - stock[u, k])

        # Subtract shipments scheduled to arrive at the sink within the planning window.
        pe = np.asarray(obs.get("pipeline.edge", [])).ravel()
        pk = np.asarray(obs.get("pipeline.k", [])).ravel()
        pq = np.asarray(obs.get("pipeline.qty", [])).ravel()
        pa = np.asarray(obs.get("pipeline.arrival_week", [])).ravel()
        live = np.asarray(obs.get("pipeline.qty.observed", np.ones(len(pq)))).ravel()
        for i in range(min(len(pe), len(pk), len(pq), len(pa))):
            if i < len(live) and not live[i]:
                continue
            e, k = int(pe[i]), int(pk[i])
            if 0 <= e < len(heads) and 0 <= k < self.nk and int(pa[i]) <= week + 3:
                key = (int(heads[e]), k)
                if key in need:
                    need[key] = max(0.0, need[key] - max(0.0, self.num(pq[i])))

        cap_now = np.asarray(obs.get("graph_now.u", [])).ravel()
        cost_now = np.asarray(obs.get("graph_now.c", [])).ravel()
        tau_now = np.asarray(obs.get("graph_now.tau", [])).ravel()
        tariff = np.asarray(obs.get("graph_now.tariff", []), dtype=float)
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
                        re = lane_edges[int(lane)]
                        if re:
                            route = [int(x) for x in re]
                    except Exception:
                        pass
                if not route or any(re < 0 or re >= len(tails) for re in route):
                    continue
                cap, weight, tau_total = np.inf, 0.0, 0.0
                for re in route:
                    if re < len(cap_now) and np.isfinite(cap_now[re]):
                        cap = min(cap, max(0.0, float(cap_now[re])))
                    c = self.num(cost_now[re]) if re < len(cost_now) else self.num(self.edges.get("c0", [0] * len(tails))[re])
                    tau = self.num(tau_now[re], 1.0) if re < len(tau_now) else 1.0
                    weight += max(0.0, c) + max(0.0, tau) * 0.15
                    tau_total += max(0.0, tau)
                    if tariff.ndim == 2 and re < tariff.shape[0] and k < tariff.shape[1]:
                        value = self.static.get("commodities", {}).get("v", [])
                        customs = self.num(value[k]) if k < len(value) else 0.0
                        weight += max(0.0, self.num(tariff[re, k])) * max(0.0, customs)
                cap = min(cap, *(edge_left[re] for re in route))
                if cap <= 1e-9:
                    continue
                u, v = int(tails[route[0]]), int(heads[route[-1]])
                arcs.append((s, route, u, v, cap, weight, tau_total))

            inf = 1e15
            dist = np.full((self.nn, self.nn), inf)
            first = np.full((self.nn, self.nn), -1, dtype=int)
            np.fill_diagonal(dist, 0.0)
            for ai, a in enumerate(arcs):
                _, _, u, v, cap, w, _ = a
                if cap > 1e-9 and w < dist[u, v]:
                    dist[u, v], first[u, v] = w, ai
            for mid in range(self.nn):
                for u in range(self.nn):
                    if dist[u, mid] >= inf:
                        continue
                    cand = dist[u, mid] + dist[mid]
                    better = cand < dist[u] - 1e-10
                    dist[u, better] = cand[better]
                    first[u, better] = first[u, mid]

            sinks = []
            for i, (u, kk) in enumerate(zip(snodes, sks)):
                d = need.get((u, kk), 0.0)
                if kk == k and d > 1e-8:
                    pi = max(0.0, self.num(penalties[i]) if i < len(penalties) else 1.0)
                    sinks.append((u, d, pi))
            sinks.sort(key=lambda x: -x[2])

            for sink, remaining, _ in sinks:
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
                            # Favor routes with lower cost and, for similar cost, less delay.
                            choices.append((dist[origin, sink] + 0.03 * a[6], ai, available, capacity))
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
                    arcs[ai] = (s, route, source, arcs[ai][3], max(0.0, arcs[ai][4] - qty), arcs[ai][5], arcs[ai][6])
                    remaining -= qty
                    need[(sink, k)] = max(0.0, remaining)

        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}
