# -2.1078793528609596
import numpy as np
import heapq


class Agent:
    def __init__(self, config=None):
        self.static = config["static"]
        self.layout = config["layout"]
        self.edges = self.static["edges"]
        self.slots = self.static["action_slots"]
        self.lanes = self.static.get("lanes", {}).get("edges", [])
        self.nslot = len(self.slots["edge"])
        self.nn = len(self.static["nodes"]["id"])
        self.nk = len(self.static["commodities"]["id"])
        self.nedge = len(self.edges["tail"])
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
        ns, nn, nk = self.nslot, self.nn, self.nk
        flows = np.zeros(ns, dtype=float)
        tails, heads = self.edges["tail"], self.edges["head"]
        slot_edges = self.slots["edge"]
        slot_ks = self.slots["k"]
        slot_lanes = self.slots.get("lane", [None] * ns)
        week = int(np.asarray(obs.get("week", [1])).ravel()[0])

        stock = np.zeros((nn, nk))
        sq = np.asarray(obs.get("stock.qty", [])).ravel()
        for i, pair in enumerate(self.layout.get("stock_slots", [])):
            if i < len(sq):
                node, k = map(int, pair)
                stock[node, k] += max(0.0, self.val(sq[i]))
        supply = np.zeros_like(stock)
        av = np.asarray(obs.get("graph_now.supply.avail", [])).ravel()
        for i, pair in enumerate(self.layout.get("supply_slots", [])):
            if i < len(av):
                node, k = map(int, pair)
                supply[node, k] += max(0.0, self.val(av[i]))

        sinks = self.static.get("sinks", {})
        snodes = [int(x) for x in sinks.get("node", [])]
        sks = [int(x) for x in sinks.get("k", [])]
        penalties = sinks.get("pi", [1.0] * len(snodes))
        forecast = np.asarray(obs.get("demand_forecast.qty", []), dtype=float)
        backlog = np.asarray(obs.get("backlog.qty", [])).ravel()
        needs = {}
        for i, (node, k) in enumerate(zip(snodes, sks)):
            # Cover a replenishment window, but use less of the forecast near
            # the horizon where dispatches may not arrive in time.
            f = 0.0
            if forecast.ndim == 2 and i < forecast.shape[0]:
                row = forecast[i]
                f = sum(max(0.0, self.val(x)) for x in row[:min(4, len(row))])
            b = self.val(backlog[i]) if i < len(backlog) else 0.0
            needs[(node, k)] = max(0.0, b + f - stock[node, k])

        # Credit inbound shipments whose current edge ends at a demand sink.
        pe = np.asarray(obs.get("pipeline.edge", [])).ravel()
        pk = np.asarray(obs.get("pipeline.k", [])).ravel()
        pq = np.asarray(obs.get("pipeline.qty", [])).ravel()
        pa = np.asarray(obs.get("pipeline.arrival_week", [])).ravel()
        live = np.asarray(obs.get("pipeline.qty.observed", np.ones(len(pq)))).ravel()
        for i in range(min(len(pe), len(pk), len(pq), len(pa))):
            if i < len(live) and not live[i]:
                continue
            e, k = int(pe[i]), int(pk[i])
            if 0 <= e < self.nedge and 0 <= k < nk:
                key = (int(heads[e]), k)
                if key in needs and int(pa[i]) <= week + 6:
                    needs[key] = max(0.0, needs[key] - max(0.0, self.val(pq[i])))

        cap_now = np.asarray(obs.get("graph_now.u", [])).ravel()
        cost_now = np.asarray(obs.get("graph_now.c", [])).ravel()
        tau_now = np.asarray(obs.get("graph_now.tau", [])).ravel()
        tariff = np.asarray(obs.get("graph_now.tariff", []), dtype=float)
        values = self.static.get("commodities", {}).get("v", [])
        mask = np.asarray(obs.get("action_mask", np.ones(ns))).ravel()
        mask_seen = bool(np.asarray(obs.get("action_mask.observed", [1])).ravel()[0])
        prohibited = np.asarray(obs.get("slot_mask", np.zeros(ns))).ravel()
        edge_left = np.full(self.nedge, np.inf)
        for e in range(self.nedge):
            if e < len(cap_now) and np.isfinite(cap_now[e]):
                edge_left[e] = max(0.0, float(cap_now[e]))

        for k in range(nk):
            # Each action slot is represented as an arc from its route's start
            # to end; sending now consumes capacity on every route edge.
            arcs = []
            adj = [[] for _ in range(nn)]
            for s in range(ns):
                if int(slot_ks[s]) != k:
                    continue
                allowed = mask[s] > 0.5 if mask_seen and s < len(mask) else not (s < len(prohibited) and prohibited[s] > 0.5)
                if not allowed:
                    continue
                e = int(slot_edges[s])
                route = [e]
                lane = slot_lanes[s] if s < len(slot_lanes) else None
                if lane is not None:
                    try:
                        le = self.lanes[int(lane)]
                        if le:
                            route = [int(x) for x in le]
                    except Exception:
                        pass
                if not route or any(re < 0 or re >= self.nedge for re in route):
                    continue
                cap, cost, tau = np.inf, 0.0, 0.0
                for re in route:
                    if re < len(cap_now) and np.isfinite(cap_now[re]):
                        cap = min(cap, max(0.0, float(cap_now[re])))
                    c = self.val(cost_now[re]) if re < len(cost_now) else 0.0
                    t = self.val(tau_now[re], 1.0) if re < len(tau_now) else 1.0
                    cost += max(0.0, c)
                    tau += max(0.0, t)
                    if tariff.ndim == 2 and re < tariff.shape[0] and k < tariff.shape[1]:
                        v = self.val(values[k]) if k < len(values) else 0.0
                        cost += max(0.0, self.val(tariff[re, k])) * max(0.0, v)
                cap = min(cap, *(edge_left[re] for re in route))
                if cap <= 1e-9:
                    continue
                u, v = int(tails[route[0]]), int(heads[route[-1]])
                # Freight dominates modest time preference; discourage routes
                # that cannot plausibly serve the current replenishment window.
                weight = cost + 0.05 * tau
                ai = len(arcs)
                arcs.append([s, route, u, v, cap, weight, tau])
                adj[u].append(ai)

            # Prioritize costly-to-miss sinks; allocate cheapest reachable
            # inventory/supply, replanning after each capacity consumption.
            order = [i for i, kk in enumerate(sks) if int(kk) == k and needs.get((snodes[i], k), 0.0) > 1e-8]
            order.sort(key=lambda i: -max(0.0, self.val(penalties[i]) if i < len(penalties) else 1.0))
            for si in order:
                sink = snodes[si]
                remaining = needs[(sink, k)]
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
                            d, u = heapq.heappop(heap)
                            if d != dist[u]:
                                continue
                            if u == sink:
                                break
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
                        if ai >= 0 and np.isfinite(dist[sink]):
                            cand = (dist[sink], origin, ai, available)
                            if best is None or cand[:2] < best[:2]:
                                best = cand
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
                needs[(sink, k)] = remaining

        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}