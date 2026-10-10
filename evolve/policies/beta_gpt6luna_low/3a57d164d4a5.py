# -2.1078633567229272
import numpy as np


class Agent:
    def __init__(self, config=None):
        self.static = config["static"]
        self.layout = config["layout"]
        self.edges = self.static["edges"]
        self.slots = self.static["action_slots"]
        self.lane_edges = self.static.get("lanes", {}).get("edges", [])
        self.nnode = len(self.static["nodes"]["id"])
        self.nk = len(self.static["commodities"]["id"])
        self.nedge = len(self.edges["tail"])
        self.nslot = len(self.slots["edge"])
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
        n, nk, ns, ne = self.nnode, self.nk, self.nslot, self.nedge
        tails, heads = self.edges["tail"], self.edges["head"]
        flows = np.zeros(ns, dtype=float)
        week_arr = np.asarray(obs.get("week", [1])).ravel()
        week = int(week_arr[0]) if len(week_arr) else 1

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
        penalties = sinks.get("pi", [1.0] * len(sn))
        backlog = np.asarray(obs.get("backlog.qty", [])).ravel()
        forecast = np.asarray(obs.get("demand_forecast.qty", []), dtype=float)
        # Plan a few weeks ahead to account for transport lead time, without
        # treating the full forecast horizon as a commitment to ship.
        need = {}
        for i, (u, k) in enumerate(zip(sn, sk)):
            b = max(0.0, self.val(backlog[i])) if i < len(backlog) else 0.0
            h = min(4, forecast.shape[1]) if forecast.ndim == 2 and i < forecast.shape[0] else 0
            f = sum(max(0.0, self.val(x)) for x in forecast[i, :h]) if h else 0.0
            need[(u, k)] = max(0.0, b + f - stock[u, k])

        # Subtract live cargo due at the sink during the planning window.
        pe = np.asarray(obs.get("pipeline.edge", [])).ravel()
        pk = np.asarray(obs.get("pipeline.k", [])).ravel()
        pq = np.asarray(obs.get("pipeline.qty", [])).ravel()
        pa = np.asarray(obs.get("pipeline.arrival_week", [])).ravel()
        live = np.asarray(obs.get("pipeline.qty.observed", np.ones(len(pq)))).ravel()
        for i in range(min(len(pe), len(pk), len(pq), len(pa))):
            if i < len(live) and not live[i]:
                continue
            e, k = int(pe[i]), int(pk[i])
            if 0 <= e < ne and 0 <= k < nk and int(pa[i]) <= week + 4:
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
        prohibited = np.asarray(obs.get("slot_mask", np.zeros(ns))).ravel()
        opened = np.asarray(obs.get("graph_now.open", [])).ravel()
        cp_index = {int(node): i for i, node in enumerate(self.layout.get("chokepoints", []))}
        edge_left = np.full(ne, np.inf)
        for e in range(min(ne, len(cap_now))):
            if np.isfinite(cap_now[e]):
                edge_left[e] = max(0.0, float(cap_now[e]))

        slot_edge = self.slots["edge"]
        slot_k = self.slots["k"]
        slot_lane = self.slots.get("lane", [None] * ns)
        for k in range(nk):
            arcs = []
            for s in range(ns):
                if int(slot_k[s]) != k:
                    continue
                allowed = (s < len(mask) and mask[s] > 0.5) if mask_seen else not (s < len(prohibited) and prohibited[s] > 0.5)
                if not allowed:
                    continue
                route = [int(slot_edge[s])]
                lane = slot_lane[s] if s < len(slot_lane) else None
                if lane is not None:
                    try:
                        le = self.lane_edges[int(lane)]
                        if le:
                            route = [int(e) for e in le]
                    except (TypeError, ValueError, IndexError):
                        pass
                if not route or any(e < 0 or e >= ne for e in route):
                    continue
                cap = min(edge_left[e] for e in route)
                weight = 0.0
                tau_sum = 0.0
                for e in route:
                    if e < len(cap_now) and np.isfinite(cap_now[e]):
                        cap = min(cap, max(0.0, float(cap_now[e])))
                    c = self.val(cost_now[e]) if e < len(cost_now) else 0.0
                    t = max(0.0, self.val(tau_now[e], 1.0)) if e < len(tau_now) else 1.0
                    weight += max(0.0, c) + 0.12 * t
                    tau_sum += t
                    if tariff.ndim == 2 and e < tariff.shape[0] and k < tariff.shape[1]:
                        v = self.val(values[k]) if k < len(values) else 0.0
                        weight += max(0.0, self.val(tariff[e, k])) * max(0.0, v)
                    cp = cp_index.get(int(heads[e]))
                    if cp is not None and cp < len(opened):
                        cap *= max(0.0, min(1.0, self.val(opened[cp])))
                if cap <= 1e-9:
                    continue
                arcs.append([s, route, int(tails[route[0]]), int(heads[route[-1]]), cap, weight, tau_sum])

            # Rebuild shortest paths after capacity is consumed so alternatives
            # can be used when a shared bottleneck fills.
            for si in sorted((i for i in range(len(sn)) if sk[i] == k and need.get((sn[i], k), 0.0) > 1e-8),
                             key=lambda i: -max(0.0, self.val(penalties[i] if i < len(penalties) else 1.0))):
                sink = sn[si]
                remaining = need[(sink, k)]
                while remaining > 1e-8:
                    inf = 1e15
                    dist = np.full(n, inf)
                    first = np.full(n, -1, dtype=int)
                    dist[sink] = 0.0
                    # Reverse Bellman-Ford over the currently available action arcs.
                    for _ in range(max(0, n - 1)):
                        changed = False
                        for ai, a in enumerate(arcs):
                            if a[4] <= 1e-9 or dist[a[3]] >= inf:
                                continue
                            nd = a[5] + dist[a[3]]
                            if nd < dist[a[2]] - 1e-10:
                                dist[a[2]] = nd
                                first[a[2]] = ai
                                changed = True
                        if not changed:
                            break
                    best = None
                    for origin in range(n):
                        available = stock[origin, k] + supply[origin, k]
                        ai = int(first[origin])
                        if origin == sink or available <= 1e-8 or ai < 0 or dist[origin] >= inf:
                            continue
                        a = arcs[ai]
                        cap = min(a[4], *(edge_left[e] for e in a[1]))
                        if cap <= 1e-8:
                            continue
                        # Scale the route cost by shortage penalty, preserving a
                        # preference for cheaper sources for valuable demand.
                        pi = max(1e-6, self.val(penalties[si] if si < len(penalties) else 1.0))
                        choice = (dist[origin] / pi, dist[origin], -available, origin, ai, cap)
                        if best is None or choice[:4] < best[:4]:
                            best = choice
                    if best is None:
                        break
                    _, _, negavail, origin, ai, cap = best
                    a = arcs[ai]
                    qty = min(remaining, -negavail, cap)
                    if qty <= 1e-8:
                        break
                    flows[a[0]] += qty
                    used = min(qty, stock[origin, k])
                    stock[origin, k] -= used
                    supply[origin, k] = max(0.0, supply[origin, k] - (qty - used))
                    for e in a[1]:
                        edge_left[e] = max(0.0, edge_left[e] - qty)
                    a[4] = max(0.0, a[4] - qty)
                    remaining -= qty
                need[(sink, k)] = remaining

        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}
