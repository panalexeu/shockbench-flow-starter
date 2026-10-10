# -2.109459066232265
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
    def num(x, default=0.0):
        try:
            y = float(x)
            return y if np.isfinite(y) else default
        except Exception:
            return default

    def act(self, obs):
        ns, nn, nk = self.nslot, self.nn, self.nk
        flows = np.zeros(ns, dtype=float)
        week = int(np.asarray(obs.get("week", [1])).ravel()[0])
        tails, heads = self.edges["tail"], self.edges["head"]
        lane_edges = self.static.get("lanes", {}).get("edges", [])
        slot_e = self.slots["edge"]
        slot_k = self.slots["k"]
        slot_lane = self.slots.get("lane", [None] * ns)

        stock = np.zeros((nn, nk))
        q = np.asarray(obs.get("stock.qty", [])).ravel()
        for i, pair in enumerate(self.layout.get("stock_slots", [])):
            if i < len(q):
                node, k = map(int, pair)
                stock[node, k] += max(0.0, self.num(q[i]))
        supply = np.zeros((nn, nk))
        av = np.asarray(obs.get("graph_now.supply.avail", [])).ravel()
        for i, pair in enumerate(self.layout.get("supply_slots", [])):
            if i < len(av):
                node, k = map(int, pair)
                supply[node, k] += max(0.0, self.num(av[i]))

        forecast = np.asarray(obs.get("demand_forecast.qty", []), dtype=float)
        backlog = np.asarray(obs.get("backlog.qty", [])).ravel()
        sinks = self.static.get("sinks", {})
        sink_nodes, sink_ks = sinks.get("node", []), sinks.get("k", [])
        sink_pi = sinks.get("pi", [1.0] * len(sink_nodes))
        need = {}
        for i, (node, k) in enumerate(zip(sink_nodes, sink_ks)):
            node, k = int(node), int(k)
            # Cover a near-term window without accumulating a long stockpile.
            f = 0.0
            if forecast.ndim == 2 and i < forecast.shape[0]:
                f = sum(max(0.0, self.num(x)) for x in forecast[i, :min(3, forecast.shape[1])])
            b = max(0.0, self.num(backlog[i])) if i < len(backlog) else 0.0
            need[(node, k)] = max(0.0, b + f - stock[node, k])

        # Account for live pipeline cargo arriving at the sink within the
        # near-term planning window; its edge head is its current destination.
        pe = np.asarray(obs.get("pipeline.edge", [])).ravel()
        pk = np.asarray(obs.get("pipeline.k", [])).ravel()
        pq = np.asarray(obs.get("pipeline.qty", [])).ravel()
        pa = np.asarray(obs.get("pipeline.arrival_week", [])).ravel()
        live = np.asarray(obs.get("pipeline.qty.observed", np.ones(len(pq)))).ravel()
        for i in range(min(len(pe), len(pk), len(pq), len(pa))):
            if i < len(live) and not live[i]:
                continue
            e, k = int(pe[i]), int(pk[i])
            if 0 <= e < len(heads) and 0 <= k < nk and int(pa[i]) <= week + 3:
                key = (int(heads[e]), k)
                if key in need:
                    need[key] = max(0.0, need[key] - max(0.0, self.num(pq[i])))

        cap_now = np.asarray(obs.get("graph_now.u", [])).ravel()
        cost_now = np.asarray(obs.get("graph_now.c", [])).ravel()
        tau_now = np.asarray(obs.get("graph_now.tau", [])).ravel()
        mask = np.asarray(obs.get("action_mask", np.ones(ns))).ravel()
        mask_observed = bool(np.asarray(obs.get("action_mask.observed", [1])).ravel()[0])
        prohibited = np.asarray(obs.get("slot_mask", np.zeros(ns))).ravel()
        edge_left = np.full(len(tails), np.inf)
        for e in range(len(tails)):
            if e < len(cap_now) and np.isfinite(cap_now[e]):
                edge_left[e] = max(0.0, float(cap_now[e]))

        # Process commodities separately, sharing edge capacity across all
        # routes of that commodity for this week's dispatch.
        for k in range(nk):
            arcs = []
            for s in range(ns):
                if int(slot_k[s]) != k:
                    continue
                allowed = (s < len(mask) and mask[s] > 0.5) if mask_observed else not (s < len(prohibited) and prohibited[s] > 0.5)
                if not allowed:
                    continue
                e = int(slot_e[s])
                route = [e]
                lane = slot_lane[s] if s < len(slot_lane) else None
                if lane is not None:
                    try:
                        li = int(lane)
                        if 0 <= li < len(lane_edges) and lane_edges[li]:
                            route = [int(x) for x in lane_edges[li]]
                    except Exception:
                        pass
                cap, cost, tau = np.inf, 0.0, 0.0
                for re in route:
                    if re < len(cap_now) and np.isfinite(cap_now[re]):
                        cap = min(cap, max(0.0, float(cap_now[re])))
                    if re < len(cost_now):
                        cost += max(0.0, self.num(cost_now[re]))
                    tau += max(0.0, self.num(tau_now[re], 1.0)) if re < len(tau_now) else 1.0
                cap = min(cap, edge_left[route[0]])
                if cap <= 1e-9:
                    continue
                u, v = int(tails[route[0]]), int(heads[route[-1]])
                arcs.append((s, route, u, v, cap, cost + 0.05 * tau, tau))

            # Recompute a shortest usable route for each sink. A route's first
            # action arc is dispatched now; subsequent legs can be dispatched
            # from their own stock when the cargo arrives.
            inf = 1e15
            dist = np.full((nn, nn), inf)
            first = np.full((nn, nn), -1, dtype=int)
            np.fill_diagonal(dist, 0.0)
            for ai, arc in enumerate(arcs):
                _, _, u, v, cap, weight, _ = arc
                if cap > 1e-9 and weight < dist[u, v]:
                    dist[u, v], first[u, v] = weight, ai
            for mid in range(nn):
                for u in range(nn):
                    if dist[u, mid] >= inf:
                        continue
                    cand = dist[u, mid] + dist[mid]
                    better = cand < dist[u] - 1e-10
                    dist[u, better] = cand[better]
                    first[u, better] = first[u, mid]

            demand_list = []
            for i, (node, kk) in enumerate(zip(sink_nodes, sink_ks)):
                node, kk = int(node), int(kk)
                d = need.get((node, kk), 0.0)
                if kk == k and d > 1e-8:
                    pi = max(0.0, self.num(sink_pi[i] if i < len(sink_pi) else 1.0, 1.0))
                    demand_list.append((node, d, pi))
            demand_list.sort(key=lambda x: -x[2])

            for sink, remaining, _ in demand_list:
                while remaining > 1e-8:
                    choices = []
                    for origin in range(nn):
                        if origin == sink or dist[origin, sink] >= inf:
                            continue
                        ai = int(first[origin, sink])
                        if ai < 0:
                            continue
                        arc = arcs[ai]
                        u = arc[2]
                        available = stock[u, k] + supply[u, k]
                        cap = min(arc[4], edge_left[arc[1][0]])
                        if available > 1e-8 and cap > 1e-8:
                            choices.append((dist[origin, sink], origin, ai, available, cap))
                    if not choices:
                        break
                    _, _, ai, available, cap = min(choices)
                    s, route, u, _, _, _, _ = arcs[ai]
                    qty = min(remaining, available, cap)
                    if qty <= 1e-8:
                        break
                    flows[s] += qty
                    from_stock = min(qty, stock[u, k])
                    stock[u, k] -= from_stock
                    supply[u, k] = max(0.0, supply[u, k] - (qty - from_stock))
                    for re in route:
                        edge_left[re] = max(0.0, edge_left[re] - qty)
                    remaining -= qty
                    need[(sink, k)] = max(0.0, need[(sink, k)] - qty)

        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}
