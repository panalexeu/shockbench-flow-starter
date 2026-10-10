# -2.114381008639432
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
            x = float(x)
            return x if np.isfinite(x) else default
        except Exception:
            return default

    def act(self, obs):
        ns, nn, nk = self.nslot, self.nn, self.nk
        flows = np.zeros(ns, dtype=float)
        tails = self.edges["tail"]
        heads = self.edges["head"]
        lane_edges = self.static.get("lanes", {}).get("edges", [])
        slot_edges = self.slots["edge"]
        slot_ks = self.slots["k"]
        slot_lanes = self.slots.get("lane", [None] * ns)
        week = int(np.asarray(obs.get("week", [1])).ravel()[0])

        stock = np.zeros((nn, nk), dtype=float)
        qstock = np.asarray(obs.get("stock.qty", [])).ravel()
        for i, pair in enumerate(self.layout.get("stock_slots", [])):
            if i < len(qstock):
                node, k = map(int, pair)
                stock[node, k] += max(0.0, self.val(qstock[i]))
        supply = np.zeros((nn, nk), dtype=float)
        av = np.asarray(obs.get("graph_now.supply.avail", [])).ravel()
        for i, pair in enumerate(self.layout.get("supply_slots", [])):
            if i < len(av):
                node, k = map(int, pair)
                supply[node, k] += max(0.0, self.val(av[i]))

        forecast = np.asarray(obs.get("demand_forecast.qty", []), dtype=float)
        backlog = np.asarray(obs.get("backlog.qty", [])).ravel()
        sink_nodes = self.static.get("sinks", {}).get("node", [])
        sink_ks = self.static.get("sinks", {}).get("k", [])
        sink_pi = self.static.get("sinks", {}).get("pi", [1.0] * len(sink_nodes))
        needs = {}
        for i, (node, k) in enumerate(zip(sink_nodes, sink_ks)):
            node, k = int(node), int(k)
            # Cover forecast demand over a useful replenishment window. Longer
            # routes justify using more of the available forecast horizon.
            max_tau = max([self.val(x, 1.0) for x in self.edges.get("tau0", [])] or [1.0])
            horizon = min(forecast.shape[1] if forecast.ndim == 2 else 1,
                          max(3, int(np.ceil(max_tau)) + 2))
            f = sum(max(0.0, self.val(x)) for x in forecast[i, :horizon]) if forecast.ndim == 2 and i < forecast.shape[0] else 0.0
            b = max(0.0, self.val(backlog[i])) if i < len(backlog) else 0.0
            needs[(node, k)] = max(0.0, b + f - stock[node, k])

        # Subtract all observed pipeline cargo that will reach the sink soon
        # enough to contribute to the forecast window.
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
                node = int(heads[e])
                key = (node, k)
                if key in needs and int(pa[i]) <= week + 8:
                    needs[key] = max(0.0, needs[key] - max(0.0, self.val(pq[i])))

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
                if int(slot_ks[s]) != k:
                    continue
                allowed = (s < len(mask) and mask[s] > 0.5) if mask_seen else not (s < len(prohibited) and prohibited[s] > 0.5)
                if not allowed:
                    continue
                e = int(slot_edges[s])
                route = [e]
                lane = slot_lanes[s] if s < len(slot_lanes) else None
                if lane is not None:
                    try:
                        li = int(lane)
                        if 0 <= li < len(lane_edges) and lane_edges[li]:
                            route = [int(x) for x in lane_edges[li]]
                    except Exception:
                        pass
                if not route:
                    continue
                cap, cost, tau = np.inf, 0.0, 0.0
                for re in route:
                    if re < len(cap_now) and np.isfinite(cap_now[re]):
                        cap = min(cap, max(0.0, float(cap_now[re])))
                    cost += max(0.0, self.val(cost_now[re] if re < len(cost_now) else self.edges.get("c0", [0] * len(tails))[re]))
                    tau += max(0.0, self.val(tau_now[re] if re < len(tau_now) else 1.0))
                cap = min(cap, edge_left[route[0]])
                if cap <= 1e-9:
                    continue
                # Freight is primary; lead time breaks close-cost ties.
                weight = cost + 0.05 * tau
                arcs.append((s, route, int(tails[route[0]]), int(heads[route[-1]]), cap, weight))

            inf = 1e15
            dist = np.full((nn, nn), inf)
            first = np.full((nn, nn), -1, dtype=int)
            np.fill_diagonal(dist, 0.0)
            for ai, arc in enumerate(arcs):
                _, _, u, v, cap, weight = arc
                if cap > 1e-9 and weight < dist[u, v]:
                    dist[u, v], first[u, v] = weight, ai
            for mid in range(nn):
                for u in range(nn):
                    if dist[u, mid] >= inf:
                        continue
                    cand = dist[u, mid] + dist[mid]
                    better = cand < dist[u] - 1e-9
                    dist[u, better] = cand[better]
                    first[u, better] = first[u, mid]

            sink_order = [(node, need, self.val(sink_pi[i], 1.0))
                          for i, ((node, kk), need) in enumerate(needs.items())
                          if kk == k and need > 1e-8]
            sink_order.sort(key=lambda x: -x[2])
            for sink, need, _ in sink_order:
                while need > 1e-8:
                    candidates = []
                    for origin in range(nn):
                        if origin == sink or dist[origin, sink] >= inf:
                            continue
                        ai = int(first[origin, sink])
                        if ai < 0:
                            continue
                        _, _, u, _, cap, _ = arcs[ai]
                        available = stock[u, k] + supply[u, k]
                        if available > 1e-8 and cap > 1e-8:
                            candidates.append((dist[origin, sink], origin, ai, available))
                    if not candidates:
                        break
                    _, _, ai, available = min(candidates)
                    s, route, u, _, cap, _ = arcs[ai]
                    qty = min(need, available, cap)
                    if qty <= 1e-8:
                        break
                    flows[s] += qty
                    take = min(qty, stock[u, k])
                    stock[u, k] -= take
                    supply[u, k] = max(0.0, supply[u, k] - (qty - take))
                    for re in route:
                        edge_left[re] = max(0.0, edge_left[re] - qty)
                    need -= qty
        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}
