# -999
# error: 'tuple' object does not support item assignment
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
            x = float(x)
            return x if np.isfinite(x) else default
        except Exception:
            return default

    def act(self, obs):
        ns, nn, nk = self.nslot, self.nn, self.nk
        flows = np.zeros(ns, dtype=float)
        tails = self.edges["tail"]
        heads = self.edges["head"]
        slot_edge = self.slots["edge"]
        slot_k = self.slots["k"]
        slot_lane = self.slots.get("lane", [None] * ns)
        lane_edges = self.static.get("lanes", {}).get("edges", [])
        week = int(np.asarray(obs.get("week", [1])).ravel()[0])

        stock = np.zeros((nn, nk), dtype=float)
        stock_obs = np.asarray(obs.get("stock.qty", [])).ravel()
        for i, pair in enumerate(self.layout.get("stock_slots", [])):
            if i < len(stock_obs):
                node, k = map(int, pair)
                stock[node, k] += max(0.0, self.num(stock_obs[i]))

        supply = np.zeros((nn, nk), dtype=float)
        avail = np.asarray(obs.get("graph_now.supply.avail", [])).ravel()
        for i, pair in enumerate(self.layout.get("supply_slots", [])):
            if i < len(avail):
                node, k = map(int, pair)
                supply[node, k] += max(0.0, self.num(avail[i]))

        sinks = self.static.get("sinks", {})
        sink_nodes = [int(x) for x in sinks.get("node", [])]
        sink_ks = [int(x) for x in sinks.get("k", [])]
        penalties = sinks.get("pi", [1.0] * len(sink_nodes))
        forecast = np.asarray(obs.get("demand_forecast.qty", []), dtype=float)
        backlog = np.asarray(obs.get("backlog.qty", [])).ravel()
        need = {}
        for i, (node, k) in enumerate(zip(sink_nodes, sink_ks)):
            # Cover the immediate forecast and backlog, with a modest buffer
            # for routes that take several weeks.
            horizon = min(3, forecast.shape[1]) if forecast.ndim == 2 and i < forecast.shape[0] else 0
            forecast_need = sum(max(0.0, self.num(x)) for x in forecast[i, :horizon]) if horizon else 0.0
            b = max(0.0, self.num(backlog[i])) if i < len(backlog) else 0.0
            need[(node, k)] = max(0.0, b + forecast_need - stock[node, k])

        # Credit shipments already on their final edge into a sink.
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
        mask_seen = bool(np.asarray(obs.get("action_mask.observed", [1])).ravel()[0])
        prohibited = np.asarray(obs.get("slot_mask", np.zeros(ns))).ravel()
        edge_left = np.full(len(tails), np.inf)
        for e in range(len(tails)):
            if e < len(cap_now) and np.isfinite(cap_now[e]):
                edge_left[e] = max(0.0, float(cap_now[e]))

        for k in range(nk):
            # Each action slot is an arc; a lane slot traverses its full lane.
            arcs = []
            for s in range(ns):
                if int(slot_k[s]) != k:
                    continue
                allowed = (s < len(mask) and mask[s] > 0.5) if mask_seen else not (s < len(prohibited) and prohibited[s] > 0.5)
                if not allowed:
                    continue
                e = int(slot_edge[s])
                route = [e]
                lane = slot_lane[s] if s < len(slot_lane) else None
                if lane is not None:
                    try:
                        le = lane_edges[int(lane)]
                        if le:
                            route = [int(x) for x in le]
                    except Exception:
                        pass
                if not route or any(re < 0 or re >= len(tails) for re in route):
                    continue
                cap, cost, tau = np.inf, 0.0, 0.0
                for re in route:
                    if re < len(cap_now) and np.isfinite(cap_now[re]):
                        cap = min(cap, max(0.0, float(cap_now[re])))
                    if re < len(cost_now):
                        cost += max(0.0, self.num(cost_now[re]))
                    tau += max(0.0, self.num(tau_now[re], 1.0)) if re < len(tau_now) else 1.0
                cap = min(cap, *(edge_left[re] for re in route))
                if cap <= 1e-9:
                    continue
                u, v = int(tails[route[0]]), int(heads[route[-1]])
                arcs.append((s, route, u, v, cap, cost + 0.1 * tau, tau))

            # All-pairs shortest paths over the currently usable action arcs.
            inf = 1e15
            dist = np.full((nn, nn), inf)
            first = np.full((nn, nn), -1, dtype=int)
            np.fill_diagonal(dist, 0.0)
            for ai, arc in enumerate(arcs):
                _, _, u, v, cap, weight, _ = arc
                if cap > 1e-9 and weight < dist[u, v]:
                    dist[u, v] = weight
                    first[u, v] = ai
            for mid in range(nn):
                cand = dist[:, mid, None] + dist[mid, None, :]
                better = cand < dist - 1e-10
                dist[better] = cand[better]
                via = np.broadcast_to(first[:, mid, None], (nn, nn))
                first[better] = via[better]

            sink_order = [i for i, kk in enumerate(sink_ks)
                          if kk == k and need.get((sink_nodes[i], k), 0.0) > 1e-8]
            sink_order.sort(key=lambda i: -max(0.0, self.num(penalties[i] if i < len(penalties) else 1.0)))
            for si in sink_order:
                sink = sink_nodes[si]
                remaining = need[(sink, k)]
                while remaining > 1e-8:
                    best = None
                    # Select the cheapest reachable currently-stocked origin.
                    for origin in range(nn):
                        if origin == sink or dist[origin, sink] >= inf:
                            continue
                        ai = int(first[origin, sink])
                        if ai < 0:
                            continue
                        arc = arcs[ai]
                        u = arc[2]
                        available = stock[u, k] + supply[u, k]
                        capacity = min(arc[4], *(edge_left[re] for re in arc[1]))
                        if available <= 1e-8 or capacity <= 1e-8:
                            continue
                        candidate = (dist[origin, sink], origin, ai, available, capacity)
                        if best is None or candidate[:2] < best[:2]:
                            best = candidate
                    if best is None:
                        break
                    _, _, ai, available, capacity = best
                    s, route, u, _, _, _, _ = arcs[ai]
                    qty = min(remaining, available, capacity)
                    if qty <= 1e-8:
                        break
                    flows[s] += qty
                    from_stock = min(qty, stock[u, k])
                    stock[u, k] -= from_stock
                    supply[u, k] = max(0.0, supply[u, k] - (qty - from_stock))
                    for re in route:
                        edge_left[re] = max(0.0, edge_left[re] - qty)
                    arcs[ai][4] = max(0.0, arcs[ai][4] - qty)
                    remaining -= qty

        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}
