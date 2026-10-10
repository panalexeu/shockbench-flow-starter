# -2.109826214799714
import numpy as np


class Agent:
    def __init__(self, config=None):
        self.static = config["static"]
        self.layout = config["layout"]
        self.edges = self.static["edges"]
        self.slots = self.static["action_slots"]
        self.n = len(self.static["nodes"]["id"])
        self.nk = len(self.static["commodities"]["id"])
        self.ns = len(self.slots["edge"])
        action = config["spaces"]["action"]
        self.override_qty = np.zeros(action["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action["release_mode"]["shape"], dtype=np.int64)

    @staticmethod
    def val(x, default=0.0):
        try:
            x = float(x)
            return x if np.isfinite(x) else default
        except Exception:
            return default

    def act(self, obs):
        flows = np.zeros(self.ns, dtype=float)
        tails, heads = self.edges["tail"], self.edges["head"]
        lane_edges = self.static.get("lanes", {}).get("edges", [])
        slot_e = self.slots["edge"]
        slot_k = self.slots["k"]
        slot_lane = self.slots.get("lane", [None] * self.ns)
        cap_now = np.asarray(obs.get("graph_now.u", [])).ravel()
        cost_now = np.asarray(obs.get("graph_now.c", [])).ravel()
        tau_now = np.asarray(obs.get("graph_now.tau", [])).ravel()
        mask = np.asarray(obs.get("action_mask", np.ones(self.ns))).ravel()
        mask_seen = bool(np.asarray(obs.get("action_mask.observed", [1])).ravel()[0])
        prohibited = np.asarray(obs.get("slot_mask", np.zeros(self.ns))).ravel()
        opened = np.asarray(obs.get("graph_now.open", [])).ravel()
        cp_index = {int(node): i for i, node in enumerate(self.layout.get("chokepoints", []))}

        stock = np.zeros((self.n, self.nk))
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

        sinks = self.static.get("sinks", {})
        snodes = [int(x) for x in sinks.get("node", [])]
        sks = [int(x) for x in sinks.get("k", [])]
        backlog = np.asarray(obs.get("backlog.qty", [])).ravel()
        forecast = np.asarray(obs.get("demand_forecast.qty", []), dtype=float)
        need = {}
        for i, (u, k) in enumerate(zip(snodes, sks)):
            f = 0.0
            if forecast.ndim == 2 and i < forecast.shape[0]:
                f = sum(max(0.0, self.val(x)) for x in forecast[i, :min(2, forecast.shape[1])])
            b = max(0.0, self.val(backlog[i])) if i < len(backlog) else 0.0
            need[(u, k)] = max(0.0, b + f - stock[u, k])

        # Credit only observed shipments that will arrive within the target window.
        week = int(np.asarray(obs.get("week", [1])).ravel()[0])
        pe = np.asarray(obs.get("pipeline.edge", [])).ravel()
        pk = np.asarray(obs.get("pipeline.k", [])).ravel()
        pq = np.asarray(obs.get("pipeline.qty", [])).ravel()
        pa = np.asarray(obs.get("pipeline.arrival_week", [])).ravel()
        live = np.asarray(obs.get("pipeline.qty.observed", np.ones(len(pq)))).ravel()
        for i in range(min(len(pe), len(pk), len(pq), len(pa))):
            if i < len(live) and not live[i]:
                continue
            e, k = int(pe[i]), int(pk[i])
            if 0 <= e < len(heads) and 0 <= k < self.nk and int(pa[i]) <= week + 2:
                key = (int(heads[e]), k)
                if key in need:
                    need[key] = max(0.0, need[key] - max(0.0, self.val(pq[i])))

        edge_left = np.full(len(tails), np.inf)
        for e in range(len(tails)):
            if e < len(cap_now) and np.isfinite(cap_now[e]):
                edge_left[e] = max(0.0, float(cap_now[e]))

        for k in range(self.nk):
            arcs = []
            for s in range(self.ns):
                if int(slot_k[s]) != k:
                    continue
                allowed = mask[s] > 0.5 if mask_seen and s < len(mask) else not (s < len(prohibited) and prohibited[s] > 0.5)
                if not allowed:
                    continue
                e = int(slot_e[s])
                route = [e]
                lane = slot_lane[s]
                if lane is not None:
                    try:
                        le = lane_edges[int(lane)]
                        if le:
                            route = [int(x) for x in le]
                    except Exception:
                        pass
                if not route:
                    continue
                cap, cost, tau, factor = np.inf, 0.0, 0.0, 1.0
                for re in route:
                    if re < len(cap_now) and np.isfinite(cap_now[re]):
                        cap = min(cap, max(0.0, float(cap_now[re])))
                    cost += max(0.0, self.val(cost_now[re])) if re < len(cost_now) else 0.0
                    tau += max(0.0, self.val(tau_now[re], 1.0)) if re < len(tau_now) else 1.0
                    cp = cp_index.get(int(heads[re]))
                    if cp is not None and cp < len(opened):
                        factor = min(factor, max(0.0, min(1.0, self.val(opened[cp]))))
                cap = min(cap, edge_left[route[0]]) * factor
                if cap <= 1e-9:
                    continue
                arcs.append((s, route, int(tails[route[0]]), int(heads[route[-1]]), cap, cost + 0.1 * tau))

            inf = 1e15
            dist = np.full((self.n, self.n), inf)
            first = np.full((self.n, self.n), -1, dtype=int)
            np.fill_diagonal(dist, 0.0)
            for ai, a in enumerate(arcs):
                _, _, u, v, cap, weight = a
                if cap > 1e-9 and weight < dist[u, v]:
                    dist[u, v], first[u, v] = weight, ai
            for mid in range(self.n):
                for u in range(self.n):
                    if dist[u, mid] >= inf:
                        continue
                    cand = dist[u, mid] + dist[mid]
                    better = cand < dist[u] - 1e-10
                    dist[u, better] = cand[better]
                    first[u, better] = first[u, mid]

            # Allocate highest-penalty demand first; for each, use the cheapest
            # reachable source with stock or this week's available supply.
            sink_indices = [i for i, (u, kk) in enumerate(zip(snodes, sks))
                            if kk == k and need.get((u, k), 0.0) > 1e-8]
            sink_indices.sort(key=lambda i: -max(0.0, self.val(sinks.get("pi", [1.0] * len(snodes))[i])))
            for i in sink_indices:
                sink = snodes[i]
                remaining = need[(sink, k)]
                while remaining > 1e-8:
                    choices = []
                    for origin in range(self.n):
                        if origin == sink or dist[origin, sink] >= inf:
                            continue
                        ai = int(first[origin, sink])
                        if ai < 0:
                            continue
                        arc = arcs[ai]
                        source = arc[2]
                        available = stock[source, k] + supply[source, k]
                        cap = min(arc[4], edge_left[arc[1][0]])
                        if available > 1e-8 and cap > 1e-8:
                            choices.append((dist[origin, sink], ai, available, cap))
                    if not choices:
                        break
                    _, ai, available, cap = min(choices)
                    s, route, source, _, _, _ = arcs[ai]
                    qty = min(remaining, available, cap)
                    if qty <= 1e-8:
                        break
                    flows[s] += qty
                    from_stock = min(qty, stock[source, k])
                    stock[source, k] -= from_stock
                    supply[source, k] = max(0.0, supply[source, k] - (qty - from_stock))
                    for re in route:
                        edge_left[re] = max(0.0, edge_left[re] - qty)
                    remaining -= qty
                    need[(sink, k)] = remaining

        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}