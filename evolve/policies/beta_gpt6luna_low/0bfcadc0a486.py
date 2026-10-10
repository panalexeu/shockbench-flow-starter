# -2.108031991680676
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
        except Exception:
            return default

    def act(self, obs):
        nn, nk, ns = self.nn, self.nk, self.ns
        flows = np.zeros(ns, dtype=float)
        tails, heads = self.edges["tail"], self.edges["head"]
        nedge = len(tails)
        stock = np.zeros((nn, nk), dtype=float)
        stock_obs = np.asarray(obs.get("stock.qty", [])).ravel()
        for i, pair in enumerate(self.layout.get("stock_slots", [])):
            if i < len(stock_obs):
                u, k = map(int, pair)
                stock[u, k] += max(0.0, self.num(stock_obs[i]))
        supply = np.zeros_like(stock)
        supply_obs = np.asarray(obs.get("graph_now.supply.avail", [])).ravel()
        for i, pair in enumerate(self.layout.get("supply_slots", [])):
            if i < len(supply_obs):
                u, k = map(int, pair)
                supply[u, k] += max(0.0, self.num(supply_obs[i]))

        sinks = self.static.get("sinks", {})
        sink_nodes = [int(x) for x in sinks.get("node", [])]
        sink_ks = [int(x) for x in sinks.get("k", [])]
        penalties = sinks.get("pi", [1.0] * len(sink_nodes))
        backlog = np.asarray(obs.get("backlog.qty", [])).ravel()
        forecast = np.asarray(obs.get("demand_forecast.qty", []), dtype=float)
        week = int(np.asarray(obs.get("week", [1])).ravel()[0])

        # A short replenishment target makes shipments useful without building
        # large speculative inventories. The route lead time is accounted for
        # in path cost below.
        need = {}
        for i, (u, k) in enumerate(zip(sink_nodes, sink_ks)):
            f = 0.0
            if forecast.ndim == 2 and i < forecast.shape[0]:
                h = min(3, forecast.shape[1])
                f = sum(max(0.0, self.num(x)) for x in forecast[i, :h])
            b = max(0.0, self.num(backlog[i])) if i < len(backlog) else 0.0
            need[(u, k)] = max(0.0, b + f - stock[u, k])

        # Subtract shipments expected to reach their final sink in the target
        # window. Pipeline entries at intermediate nodes are not sink supply.
        pe = np.asarray(obs.get("pipeline.edge", [])).ravel()
        pk = np.asarray(obs.get("pipeline.k", [])).ravel()
        pq = np.asarray(obs.get("pipeline.qty", [])).ravel()
        pa = np.asarray(obs.get("pipeline.arrival_week", [])).ravel()
        live = np.asarray(obs.get("pipeline.qty.observed", np.ones(len(pq)))).ravel()
        for i in range(min(len(pe), len(pk), len(pq), len(pa))):
            if i < len(live) and not live[i]:
                continue
            e, k = int(pe[i]), int(pk[i])
            if 0 <= e < nedge and 0 <= k < nk and int(pa[i]) <= week + 3:
                key = (int(heads[e]), k)
                if key in need:
                    need[key] = max(0.0, need[key] - max(0.0, self.num(pq[i])))

        cap_now = np.asarray(obs.get("graph_now.u", [])).ravel()
        costs = np.asarray(obs.get("graph_now.c", [])).ravel()
        taus = np.asarray(obs.get("graph_now.tau", [])).ravel()
        tariffs = np.asarray(obs.get("graph_now.tariff", []), dtype=float)
        values = self.static.get("commodities", {}).get("v", [])
        mask = np.asarray(obs.get("action_mask", np.ones(ns))).ravel()
        mask_seen = bool(np.asarray(obs.get("action_mask.observed", [1])).ravel()[0])
        prohibited = np.asarray(obs.get("slot_mask", np.zeros(ns))).ravel()
        lane_edges = self.static.get("lanes", {}).get("edges", [])
        slot_e = self.slots["edge"]
        slot_k = self.slots["k"]
        slot_lane = self.slots.get("lane", [None] * ns)
        edge_left = np.full(nedge, np.inf)
        for e in range(nedge):
            if e < len(cap_now) and np.isfinite(cap_now[e]):
                edge_left[e] = max(0.0, float(cap_now[e]))

        for k in range(nk):
            # Construct the directed graph of available action slots. A lane
            # slot is one action that traverses its whole lane.
            arcs = []
            outgoing = [[] for _ in range(nn)]
            for s in range(ns):
                if int(slot_k[s]) != k:
                    continue
                allowed = mask_seen and s < len(mask) and mask[s] > 0.5
                if not mask_seen:
                    allowed = not (s < len(prohibited) and prohibited[s] > 0.5)
                if not allowed:
                    continue
                e = int(slot_e[s])
                route = [e]
                lane = slot_lane[s] if s < len(slot_lane) else None
                if lane is not None:
                    try:
                        r = lane_edges[int(lane)]
                        if r:
                            route = [int(x) for x in r]
                    except Exception:
                        pass
                if not route or any(x < 0 or x >= nedge for x in route):
                    continue
                u, v = int(tails[route[0]]), int(heads[route[-1]])
                cap = min(edge_left[x] for x in route)
                cost = 0.0
                tau = 0.0
                for x in route:
                    if x < len(cap_now) and np.isfinite(cap_now[x]):
                        cap = min(cap, max(0.0, float(cap_now[x])))
                    cost += max(0.0, self.num(costs[x])) if x < len(costs) else 0.0
                    tau += max(0.0, self.num(taus[x], 1.0)) if x < len(taus) else 1.0
                    if tariffs.ndim == 2 and x < tariffs.shape[0] and k < tariffs.shape[1]:
                        val = max(0.0, self.num(values[k])) if k < len(values) else 0.0
                        cost += max(0.0, self.num(tariffs[x, k])) * val
                if cap <= 1e-9:
                    continue
                # Freight plus a modest time penalty; tariffs are included in
                # freight cost so disrupted expensive routes are avoided.
                ai = len(arcs)
                arcs.append([s, route, u, v, cap, cost + 0.2 * tau])
                outgoing[u].append(ai)

            for si, (sink, kk) in sorted(
                enumerate(zip(sink_nodes, sink_ks)),
                key=lambda z: -max(0.0, self.num(penalties[z[0]]) if z[0] < len(penalties) else 1.0),
            ):
                if kk != k:
                    continue
                remaining = need.get((sink, k), 0.0)
                while remaining > 1e-8:
                    # Find the least-cost available source-to-sink path. The
                    # first action arc is dispatched this week; later arcs
                    # become decisions when the goods reach their nodes.
                    best = None
                    for origin in range(nn):
                        available = stock[origin, k] + supply[origin, k]
                        if origin == sink or available <= 1e-8:
                            continue
                        dist = [float("inf")] * nn
                        first = [-1] * nn
                        used = [False] * nn
                        dist[origin] = 0.0
                        for _ in range(nn):
                            u = -1
                            for j in range(nn):
                                if not used[j] and (u < 0 or dist[j] < dist[u]):
                                    u = j
                            if u < 0 or not np.isfinite(dist[u]):
                                break
                            if u == sink:
                                break
                            used[u] = True
                            for ai in outgoing[u]:
                                a = arcs[ai]
                                if a[4] <= 1e-9:
                                    continue
                                nd = dist[u] + a[5]
                                if nd < dist[a[3]]:
                                    dist[a[3]] = nd
                                    first[a[3]] = ai if u == origin else first[u]
                        ai = first[sink]
                        if ai < 0 or not np.isfinite(dist[sink]):
                            continue
                        a = arcs[ai]
                        source = a[2]
                        avail = stock[source, k] + supply[source, k]
                        if avail <= 1e-8:
                            continue
                        candidate = (dist[sink], origin, ai, avail)
                        if best is None or candidate[:2] < best[:2]:
                            best = candidate
                    if best is None:
                        break
                    _, _, ai, avail = best
                    a = arcs[ai]
                    s, route, source, _, _, _ = a
                    qty = min(remaining, avail, a[4], *(edge_left[x] for x in route))
                    if qty <= 1e-8:
                        break
                    flows[s] += qty
                    from_stock = min(qty, stock[source, k])
                    stock[source, k] -= from_stock
                    supply[source, k] = max(0.0, supply[source, k] - (qty - from_stock))
                    for x in route:
                        edge_left[x] = max(0.0, edge_left[x] - qty)
                    a[4] = max(0.0, a[4] - qty)
                    remaining -= qty
                    need[(sink, k)] = remaining

        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}