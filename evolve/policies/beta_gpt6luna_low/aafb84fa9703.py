# -2.108749832622179
import numpy as np


class Agent:
    def __init__(self, config=None):
        self.static = config["static"]
        self.layout = config["layout"]
        self.edges = self.static["edges"]
        self.slots = self.static["action_slots"]
        self.n_nodes = len(self.static["nodes"]["id"])
        self.n_goods = len(self.static["commodities"]["id"])
        self.n_slots = len(self.slots["edge"])
        action = config["spaces"]["action"]
        self.override_qty = np.zeros(action["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action["release_mode"]["shape"], dtype=np.int64)
        self.lane_edges = self.static.get("lanes", {}).get("edges", [])

    @staticmethod
    def val(x, default=0.0):
        try:
            y = float(x)
            return y if np.isfinite(y) else default
        except Exception:
            return default

    def act(self, obs):
        n, nk, ns = self.n_nodes, self.n_goods, self.n_slots
        flows = np.zeros(ns, dtype=float)
        tails, heads = self.edges["tail"], self.edges["head"]
        nedge = len(tails)

        stock = np.zeros((n, nk), dtype=float)
        stock_obs = np.asarray(obs.get("stock.qty", [])).ravel()
        for i, pair in enumerate(self.layout.get("stock_slots", [])):
            if i < len(stock_obs):
                u, k = map(int, pair)
                stock[u, k] += max(0.0, self.val(stock_obs[i]))

        supply = np.zeros((n, nk), dtype=float)
        avail = np.asarray(obs.get("graph_now.supply.avail", [])).ravel()
        for i, pair in enumerate(self.layout.get("supply_slots", [])):
            if i < len(avail):
                u, k = map(int, pair)
                supply[u, k] += max(0.0, self.val(avail[i]))

        sinks = self.static.get("sinks", {})
        snodes = [int(x) for x in sinks.get("node", [])]
        sks = [int(x) for x in sinks.get("k", [])]
        penalties = sinks.get("pi", [1.0] * len(snodes))
        backlog = np.asarray(obs.get("backlog.qty", [])).ravel()
        forecast = np.asarray(obs.get("demand_forecast.qty", []), dtype=float)
        week = int(np.asarray(obs.get("week", [1])).ravel()[0])

        need = {}
        for i, (u, k) in enumerate(zip(snodes, sks)):
            b = max(0.0, self.val(backlog[i])) if i < len(backlog) else 0.0
            f = 0.0
            if forecast.ndim == 2 and i < forecast.shape[0]:
                # A short replenishment window avoids treating the full horizon as immediate demand.
                f = sum(max(0.0, self.val(x)) for x in forecast[i, :min(2, forecast.shape[1])])
            need[(u, k)] = max(0.0, b + f - stock[u, k])

        # Count only shipments observed to arrive at a demand sink within the short window.
        pe = np.asarray(obs.get("pipeline.edge", [])).ravel()
        pk = np.asarray(obs.get("pipeline.k", [])).ravel()
        pq = np.asarray(obs.get("pipeline.qty", [])).ravel()
        pa = np.asarray(obs.get("pipeline.arrival_week", [])).ravel()
        live = np.asarray(obs.get("pipeline.qty.observed", np.ones(len(pq)))).ravel()
        for i in range(min(len(pe), len(pk), len(pq), len(pa))):
            if i < len(live) and not live[i]:
                continue
            e, k = int(pe[i]), int(pk[i])
            if 0 <= e < nedge and 0 <= k < nk and int(pa[i]) <= week + 2:
                key = (int(heads[e]), k)
                if key in need:
                    need[key] = max(0.0, need[key] - max(0.0, self.val(pq[i])))

        cap_now = np.asarray(obs.get("graph_now.u", [])).ravel()
        cost_now = np.asarray(obs.get("graph_now.c", [])).ravel()
        tau_now = np.asarray(obs.get("graph_now.tau", [])).ravel()
        tariff = np.asarray(obs.get("graph_now.tariff", []), dtype=float)
        values = self.static.get("commodities", {}).get("v", [])
        mask = np.asarray(obs.get("action_mask", np.ones(ns))).ravel()
        mask_seen = bool(np.asarray(obs.get("action_mask.observed", [1])).ravel()[0])
        slot_mask = np.asarray(obs.get("slot_mask", np.zeros(ns))).ravel()

        # Remaining capacity is shared by every action slot using an edge.
        edge_left = np.full(nedge, np.inf)
        for e in range(min(nedge, len(cap_now))):
            if np.isfinite(cap_now[e]):
                edge_left[e] = max(0.0, float(cap_now[e]))

        cp_pos = {int(node): i for i, node in enumerate(self.layout.get("chokepoints", []))}
        opened = np.asarray(obs.get("graph_now.open", [])).ravel()

        for k in range(nk):
            arcs = []
            for s in range(ns):
                if int(self.slots["k"][s]) != k:
                    continue
                allowed = (s < len(mask) and mask[s] > 0.5) if mask_seen else not (s < len(slot_mask) and slot_mask[s] > 0.5)
                if not allowed:
                    continue
                route = [int(self.slots["edge"][s])]
                lane = self.slots.get("lane", [None] * ns)[s]
                if lane is not None:
                    try:
                        le = self.lane_edges[int(lane)]
                        if le:
                            route = [int(e) for e in le]
                    except Exception:
                        pass
                if not route or any(e < 0 or e >= nedge for e in route):
                    continue
                cap = np.inf
                weight = 0.0
                open_factor = 1.0
                for e in route:
                    if e < len(cap_now) and np.isfinite(cap_now[e]):
                        cap = min(cap, max(0.0, float(cap_now[e])))
                    c = self.val(cost_now[e]) if e < len(cost_now) else 0.0
                    tau = self.val(tau_now[e], 1.0) if e < len(tau_now) else 1.0
                    weight += max(0.0, c) + 0.05 * max(0.0, tau)
                    if tariff.ndim == 2 and e < tariff.shape[0] and k < tariff.shape[1]:
                        v = self.val(values[k]) if k < len(values) else 0.0
                        weight += max(0.0, self.val(tariff[e, k])) * max(0.0, v)
                    cp = cp_pos.get(int(heads[e]))
                    if cp is not None and cp < len(opened):
                        open_factor = min(open_factor, max(0.0, min(1.0, self.val(opened[cp]))))
                cap = min(cap, *(edge_left[e] for e in route)) * open_factor
                if cap <= 1e-9:
                    continue
                arcs.append({"slot": s, "route": route, "u": int(tails[route[0]]),
                             "v": int(heads[route[-1]]), "cap": cap, "weight": weight})

            # Prioritize the sinks with the largest shortage penalty.
            sink_order = [i for i, kk in enumerate(sks)
                          if kk == k and need.get((snodes[i], k), 0.0) > 1e-9]
            sink_order.sort(key=lambda i: -max(0.0, self.val(penalties[i] if i < len(penalties) else 1.0)))

            for si in sink_order:
                sink = snodes[si]
                remaining = need[(sink, k)]
                while remaining > 1e-8:
                    # Find the cheapest currently stocked origin and its cheapest path.
                    best = None
                    for origin in range(n):
                        available = stock[origin, k] + supply[origin, k]
                        if origin == sink or available <= 1e-8:
                            continue
                        dist = np.full(n, np.inf)
                        first = np.full(n, -1, dtype=int)
                        used = np.zeros(n, dtype=bool)
                        dist[origin] = 0.0
                        for _ in range(n):
                            u = -1
                            for x in range(n):
                                if not used[x] and (u < 0 or dist[x] < dist[u]):
                                    u = x
                            if u < 0 or not np.isfinite(dist[u]):
                                break
                            if u == sink:
                                break
                            used[u] = True
                            for ai, arc in enumerate(arcs):
                                if arc["u"] != u or arc["cap"] <= 1e-9:
                                    continue
                                nd = dist[u] + arc["weight"]
                                if nd < dist[arc["v"]]:
                                    dist[arc["v"]] = nd
                                    first[arc["v"]] = ai if u == origin else first[u]
                        ai = int(first[sink])
                        if ai < 0 or not np.isfinite(dist[sink]):
                            continue
                        arc = arcs[ai]
                        source = arc["u"]
                        quantity = min(remaining, stock[source, k] + supply[source, k], arc["cap"])
                        candidate = (dist[sink], -available, ai, source, quantity)
                        if quantity > 1e-8 and (best is None or candidate[:2] < best[:2]):
                            best = candidate
                    if best is None:
                        break
                    _, _, ai, source, quantity = best
                    arc = arcs[ai]
                    flows[arc["slot"]] += quantity
                    from_stock = min(quantity, stock[source, k])
                    stock[source, k] -= from_stock
                    supply[source, k] = max(0.0, supply[source, k] - (quantity - from_stock))
                    for e in arc["route"]:
                        edge_left[e] = max(0.0, edge_left[e] - quantity)
                    arc["cap"] = max(0.0, arc["cap"] - quantity)
                    remaining -= quantity
                need[(sink, k)] = remaining

        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}