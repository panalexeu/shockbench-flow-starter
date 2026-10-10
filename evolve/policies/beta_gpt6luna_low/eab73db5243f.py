# -2.1078684849035882
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
    def val(x, default=0.0):
        try:
            y = float(x)
            return y if np.isfinite(y) else default
        except Exception:
            return default

    def act(self, obs):
        flows = np.zeros(self.nslot, dtype=float)
        n, nk, ns = self.nn, self.nk, self.nslot
        tails = self.edges["tail"]
        heads = self.edges["head"]
        lane_edges = self.static.get("lanes", {}).get("edges", [])
        slot_e = self.slots["edge"]
        slot_k = self.slots["k"]
        slot_lane = self.slots.get("lane", [None] * ns)

        stock = np.zeros((n, nk), dtype=float)
        q = np.asarray(obs.get("stock.qty", [])).ravel()
        for i, (node, k) in enumerate(self.layout.get("stock_slots", [])):
            if i < len(q):
                stock[int(node), int(k)] += max(0.0, self.val(q[i]))
        supply = np.zeros_like(stock)
        q = np.asarray(obs.get("graph_now.supply.avail", [])).ravel()
        for i, (node, k) in enumerate(self.layout.get("supply_slots", [])):
            if i < len(q):
                supply[int(node), int(k)] += max(0.0, self.val(q[i]))

        week = int(np.asarray(obs.get("week", [1])).ravel()[0])
        sinks = self.static.get("sinks", {})
        snodes = [int(x) for x in sinks.get("node", [])]
        sks = [int(x) for x in sinks.get("k", [])]
        penalties = sinks.get("pi", [1.0] * len(snodes))
        backlog = np.asarray(obs.get("backlog.qty", [])).ravel()
        forecast = np.asarray(obs.get("demand_forecast.qty", []), dtype=float)
        demand = {}
        for i, (node, k) in enumerate(zip(snodes, sks)):
            f = 0.0
            if forecast.ndim == 2 and i < forecast.shape[0]:
                # Forecast is indexed from this week; use a modest replenishment window.
                for x in forecast[i, :min(4, forecast.shape[1])]:
                    f += max(0.0, self.val(x))
            b = max(0.0, self.val(backlog[i])) if i < len(backlog) else 0.0
            demand[(node, k)] = max(0.0, f + b - stock[node, k])

        # Credit only cargo whose final currently-recorded edge reaches a sink.
        pe = np.asarray(obs.get("pipeline.edge", [])).ravel()
        pk = np.asarray(obs.get("pipeline.k", [])).ravel()
        pq = np.asarray(obs.get("pipeline.qty", [])).ravel()
        pa = np.asarray(obs.get("pipeline.arrival_week", [])).ravel()
        live = np.asarray(obs.get("pipeline.qty.observed", np.ones(len(pq)))).ravel()
        for i in range(min(len(pe), len(pk), len(pq), len(pa))):
            if i < len(live) and not live[i]:
                continue
            e, k = int(pe[i]), int(pk[i])
            if 0 <= e < len(heads) and 0 <= k < nk:
                key = (int(heads[e]), k)
                if key in demand and int(pa[i]) <= week + 4:
                    demand[key] = max(0.0, demand[key] - max(0.0, self.val(pq[i])))

        cap = np.asarray(obs.get("graph_now.u", [])).ravel()
        costs = np.asarray(obs.get("graph_now.c", [])).ravel()
        tau = np.asarray(obs.get("graph_now.tau", [])).ravel()
        tariff = np.asarray(obs.get("graph_now.tariff", []), dtype=float)
        commodity_value = self.static.get("commodities", {}).get("v", [])
        opened = np.asarray(obs.get("graph_now.open", [])).ravel()
        cp_pos = {int(node): i for i, node in enumerate(self.layout.get("chokepoints", []))}
        mask = np.asarray(obs.get("action_mask", np.ones(ns))).ravel()
        mask_seen = bool(np.asarray(obs.get("action_mask.observed", [1])).ravel()[0])
        prohibited = np.asarray(obs.get("slot_mask", np.zeros(ns))).ravel()
        edge_left = np.full(len(tails), np.inf)
        for e in range(len(tails)):
            if e < len(cap) and np.isfinite(cap[e]):
                edge_left[e] = max(0.0, float(cap[e]))

        for k in range(nk):
            # Build usable directed action arcs for this commodity. Each arc is
            # one actual action slot, even when that slot represents a lane.
            arcs = []
            for s in range(ns):
                if int(slot_k[s]) != k:
                    continue
                allowed = mask[s] > 0.5 if mask_seen and s < len(mask) else not (s < len(prohibited) and prohibited[s] > 0.5)
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
                valid = all(0 <= re < len(tails) for re in route)
                if not valid:
                    continue
                route_cap, weight = np.inf, 0.0
                u, v = int(tails[route[0]]), int(heads[route[-1]])
                for re in route:
                    if re < len(cap) and np.isfinite(cap[re]):
                        route_cap = min(route_cap, max(0.0, float(cap[re])))
                    c = self.val(costs[re]) if re < len(costs) else 0.0
                    t = self.val(tau[re], 1.0) if re < len(tau) else 1.0
                    weight += max(0.0, c) + 0.15 * max(0.0, t)
                    if tariff.ndim == 2 and re < tariff.shape[0] and k < tariff.shape[1]:
                        value = self.val(commodity_value[k]) if k < len(commodity_value) else 0.0
                        weight += max(0.0, self.val(tariff[re, k])) * max(0.0, value)
                    cp = cp_pos.get(int(heads[re]))
                    if cp is not None and cp < len(opened):
                        route_cap *= max(0.0, min(1.0, self.val(opened[cp])))
                route_cap = min(route_cap, *(edge_left[re] for re in route))
                if route_cap > 1e-9:
                    arcs.append([s, route, u, v, route_cap, weight])

            # Repeatedly choose a shortest currently feasible path from any
            # available origin to the highest-priority unmet sink.
            for si, (sink, kk) in sorted(enumerate(zip(snodes, sks)),
                                          key=lambda z: -max(0.0, self.val(penalties[z[0]]) if z[0] < len(penalties) else 1.0)):
                if int(kk) != k:
                    continue
                remaining = demand.get((int(sink), k), 0.0)
                while remaining > 1e-8:
                    best = None
                    # Dijkstra from each stocked/supplied origin; retain first arc.
                    for origin in range(n):
                        available = stock[origin, k] + supply[origin, k]
                        if origin == int(sink) or available <= 1e-8:
                            continue
                        dist = [float("inf")] * n
                        first = [-1] * n
                        used = [False] * n
                        dist[origin] = 0.0
                        for _ in range(n):
                            u = min((j for j in range(n) if not used[j]), key=lambda j: dist[j], default=-1)
                            if u < 0 or not np.isfinite(dist[u]):
                                break
                            used[u] = True
                            for ai, a in enumerate(arcs):
                                if a[2] != u or a[4] <= 1e-9:
                                    continue
                                nd = dist[u] + a[5]
                                if nd < dist[a[3]]:
                                    dist[a[3]] = nd
                                    first[a[3]] = ai if u == origin else first[u]
                        ai = first[int(sink)]
                        if ai >= 0 and np.isfinite(dist[int(sink)]):
                            candidate = (dist[int(sink)], origin, ai, available)
                            if best is None or candidate[0] < best[0]:
                                best = candidate
                    if best is None:
                        break
                    _, origin, ai, available = best
                    a = arcs[ai]
                    qty = min(remaining, available, a[4], *(edge_left[re] for re in a[1]))
                    if qty <= 1e-8:
                        break
                    flows[a[0]] += qty
                    from_stock = min(qty, stock[origin, k])
                    stock[origin, k] -= from_stock
                    supply[origin, k] = max(0.0, supply[origin, k] - (qty - from_stock))
                    for re in a[1]:
                        edge_left[re] = max(0.0, edge_left[re] - qty)
                    a[4] = max(0.0, a[4] - qty)
                    remaining -= qty
                    demand[(int(sink), k)] = remaining

        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}