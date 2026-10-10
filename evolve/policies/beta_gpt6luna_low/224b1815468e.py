# -2.1098290908941038
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
            z = float(x)
            return z if np.isfinite(z) else default
        except Exception:
            return default

    def act(self, obs):
        n, nk, ns = self.nn, self.nk, self.nslot
        flows = np.zeros(ns, dtype=float)
        week = int(np.asarray(obs.get("week", [1])).ravel()[0])
        tails, heads = self.edges["tail"], self.edges["head"]
        lane_edges = self.static.get("lanes", {}).get("edges", [])
        slot_e = self.slots["edge"]
        slot_k = self.slots["k"]
        slot_lane = self.slots.get("lane", [None] * ns)

        stock = np.zeros((n, nk), dtype=float)
        sq = np.asarray(obs.get("stock.qty", [])).ravel()
        for i, pair in enumerate(self.layout.get("stock_slots", [])):
            if i < len(sq):
                u, k = map(int, pair)
                stock[u, k] += max(0.0, self.val(sq[i]))
        supply = np.zeros((n, nk), dtype=float)
        av = np.asarray(obs.get("graph_now.supply.avail", [])).ravel()
        for i, pair in enumerate(self.layout.get("supply_slots", [])):
            if i < len(av):
                u, k = map(int, pair)
                supply[u, k] += max(0.0, self.val(av[i]))

        sink_table = self.static.get("sinks", {})
        snodes = sink_table.get("node", [])
        sks = sink_table.get("k", [])
        pis = sink_table.get("pi", [1.0] * len(snodes))
        forecast = np.asarray(obs.get("demand_forecast.qty", []), dtype=float)
        backlog = np.asarray(obs.get("backlog.qty", [])).ravel()
        need = {}
        for i, (u, k) in enumerate(zip(snodes, sks)):
            u, k = int(u), int(k)
            # A short replenishment window limits both stockouts and over-ordering.
            f = 0.0
            if forecast.ndim == 2 and i < forecast.shape[0]:
                f = sum(max(0.0, self.val(x)) for x in forecast[i, :min(2, forecast.shape[1])])
            b = max(0.0, self.val(backlog[i])) if i < len(backlog) else 0.0
            need[(u, k)] = max(0.0, b + f - stock[u, k])

        # Credit shipments that are due to reach the demand sink soon.
        pe = np.asarray(obs.get("pipeline.edge", [])).ravel()
        pk = np.asarray(obs.get("pipeline.k", [])).ravel()
        pq = np.asarray(obs.get("pipeline.qty", [])).ravel()
        pa = np.asarray(obs.get("pipeline.arrival_week", [])).ravel()
        live = np.asarray(obs.get("pipeline.qty.observed", np.ones(len(pq)))).ravel()
        for i in range(min(len(pe), len(pk), len(pq), len(pa))):
            if i < len(live) and not live[i]:
                continue
            e, k = int(pe[i]), int(pk[i])
            if 0 <= e < len(heads) and 0 <= k < nk and int(pa[i]) <= week + 2:
                key = (int(heads[e]), k)
                if key in need:
                    need[key] = max(0.0, need[key] - max(0.0, self.val(pq[i])))

        cap_now = np.asarray(obs.get("graph_now.u", [])).ravel()
        cost_now = np.asarray(obs.get("graph_now.c", [])).ravel()
        tau_now = np.asarray(obs.get("graph_now.tau", [])).ravel()
        mask = np.asarray(obs.get("action_mask", np.ones(ns))).ravel()
        mask_seen = bool(np.asarray(obs.get("action_mask.observed", [1])).ravel()[0])
        prohibited = np.asarray(obs.get("slot_mask", np.zeros(ns))).ravel()
        edge_left = np.full(len(tails), np.inf)
        for e in range(len(tails)):
            if e < len(cap_now) and np.isfinite(cap_now[e]):
                edge_left[e] = max(0.0, float(cap_now[e]))

        for k in range(nk):
            arcs = []
            for s in range(ns):
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
                if not route:
                    continue
                cap, cost, tau = np.inf, 0.0, 0.0
                for re in route:
                    if re < len(cap_now) and np.isfinite(cap_now[re]):
                        cap = min(cap, max(0.0, float(cap_now[re])))
                    cost += max(0.0, self.val(cost_now[re])) if re < len(cost_now) else 0.0
                    tau += max(0.0, self.val(tau_now[re], 1.0)) if re < len(tau_now) else 1.0
                if cap <= 1e-9:
                    continue
                u, v = int(tails[route[0]]), int(heads[route[-1]])
                arcs.append((s, route, u, v, cap, cost + 0.15 * tau, tau))

            inf = 1e15
            dist = np.full((n, n), inf)
            first = np.full((n, n), -1, dtype=int)
            np.fill_diagonal(dist, 0.0)
            for ai, a in enumerate(arcs):
                _, _, u, v, cap, w, _ = a
                if cap > 1e-9 and w < dist[u, v]:
                    dist[u, v], first[u, v] = w, ai
            for mid in range(n):
                for u in range(n):
                    if dist[u, mid] >= inf:
                        continue
                    cand = dist[u, mid] + dist[mid]
                    better = cand < dist[u] - 1e-10
                    dist[u, better] = cand[better]
                    first[u, better] = first[u, mid]

            # Repeatedly serve the highest-value reachable deficit using its
            # cheapest available source and first dispatch leg.
            sinks = []
            for i, (u, kk) in enumerate(zip(snodes, sks)):
                u, kk = int(u), int(kk)
                d = need.get((u, kk), 0.0)
                if kk == k and d > 1e-8:
                    pi = max(0.0, self.val(pis[i] if i < len(pis) else 1.0, 1.0))
                    sinks.append((u, d, pi))
            while sinks:
                candidates = []
                for ix, (sink, d, pi) in enumerate(sinks):
                    if d <= 1e-8:
                        continue
                    for origin in range(n):
                        if origin == sink or dist[origin, sink] >= inf:
                            continue
                        ai = int(first[origin, sink])
                        if ai < 0:
                            continue
                        a = arcs[ai]
                        source = a[2]
                        available = stock[source, k] + supply[source, k]
                        cap = min(a[4], edge_left[a[1][0]])
                        if available > 1e-8 and cap > 1e-8:
                            # Prefer high shortage penalty, then economical routes.
                            candidates.append((-pi, dist[origin, sink], ix, ai, available, cap))
                if not candidates:
                    break
                _, _, ix, ai, available, cap = min(candidates)
                sink, d, pi = sinks[ix]
                s, route, source, _, _, _, _ = arcs[ai]
                qty = min(d, available, cap)
                flows[s] += qty
                take = min(qty, stock[source, k])
                stock[source, k] -= take
                supply[source, k] = max(0.0, supply[source, k] - (qty - take))
                for re in route:
                    edge_left[re] = max(0.0, edge_left[re] - qty)
                sinks[ix] = (sink, d - qty, pi)
                need[(sink, k)] = max(0.0, need[(sink, k)] - qty)

        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}
