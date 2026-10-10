# -2.108179217168013
import numpy as np


class Agent:
    def __init__(self, config=None):
        self.static = config["static"]
        self.layout = config["layout"]
        self.edges = self.static["edges"]
        self.slots = self.static["action_slots"]
        self.nslot = len(self.slots["edge"])
        self.nodes = len(self.static["nodes"]["id"])
        self.nk = len(self.static["commodities"]["id"])
        action = config["spaces"]["action"]
        self.override_qty = np.zeros(action["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action["release_mode"]["shape"], dtype=np.int64)

    @staticmethod
    def val(x):
        try:
            x = float(x)
            return max(0.0, x) if np.isfinite(x) else 0.0
        except Exception:
            return 0.0

    def act(self, obs):
        ns, n, nk = self.nslot, self.nodes, self.nk
        flows = np.zeros(ns, dtype=float)
        week = int(np.asarray(obs.get("week", [1])).ravel()[0])
        tails, heads = self.edges["tail"], self.edges["head"]
        lane_edges = self.static.get("lanes", {}).get("edges", [])
        slot_edges = self.slots["edge"]
        slot_ks = self.slots["k"]
        slot_lanes = self.slots.get("lane", [None] * ns)

        stock = np.zeros((n, nk), dtype=float)
        for i, pair in enumerate(self.layout.get("stock_slots", [])):
            q = np.asarray(obs.get("stock.qty", [])).ravel()
            if i < len(q):
                node, k = map(int, pair)
                stock[node, k] += self.val(q[i])

        supply = np.zeros((n, nk), dtype=float)
        avail = np.asarray(obs.get("graph_now.supply.avail", [])).ravel()
        for i, pair in enumerate(self.layout.get("supply_slots", [])):
            if i < len(avail):
                node, k = map(int, pair)
                supply[node, k] += self.val(avail[i])

        forecast = np.asarray(obs.get("demand_forecast.qty", []), dtype=float)
        backlog = np.asarray(obs.get("backlog.qty", [])).ravel()
        sink_nodes = self.static.get("sinks", {}).get("node", [])
        sink_ks = self.static.get("sinks", {}).get("k", [])
        sink_pi = self.static.get("sinks", {}).get("pi", [1.0] * len(sink_nodes))
        need = {}
        for i, (node, k) in enumerate(zip(sink_nodes, sink_ks)):
            near = 0.0
            if forecast.ndim == 2 and i < forecast.shape[0]:
                near = sum(self.val(x) for x in forecast[i, :min(3, forecast.shape[1])])
            b = self.val(backlog[i]) if i < len(backlog) else 0.0
            need[(int(node), int(k))] = max(0.0, b + near - stock[int(node), int(k)])

        # Subtract cargo already scheduled to reach each demand sink soon.
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
                    need[key] = max(0.0, need[key] - self.val(pq[i]))

        cap_now = np.asarray(obs.get("graph_now.u", [])).ravel()
        cost_now = np.asarray(obs.get("graph_now.c", [])).ravel()
        tau_now = np.asarray(obs.get("graph_now.tau", [])).ravel()
        cp_nodes = [int(x) for x in self.layout.get("chokepoints", [])]
        cp_pos = {node: i for i, node in enumerate(cp_nodes)}
        opened = np.asarray(obs.get("graph_now.open", [])).ravel()
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
                rcap, cost, tau, open_factor = np.inf, 0.0, 0.0, 1.0
                for re in route:
                    if re < len(cap_now) and np.isfinite(cap_now[re]):
                        rcap = min(rcap, max(0.0, float(cap_now[re])))
                    if re < len(cost_now):
                        cost += self.val(cost_now[re])
                    elif re < len(self.edges.get("c0", [])):
                        cost += self.val(self.edges["c0"][re])
                    tau += self.val(tau_now[re]) if re < len(tau_now) else 1.0
                    cp = cp_pos.get(int(heads[re]))
                    if cp is not None and cp < len(opened):
                        open_factor *= min(1.0, self.val(opened[cp]))
                rcap = min(rcap, edge_left[route[0]])
                rcap *= open_factor
                if rcap <= 1e-9:
                    continue
                u, v = int(tails[route[0]]), int(heads[route[-1]])
                arcs.append((s, route, u, v, rcap, cost + 0.1 * tau))

            inf = 1e15
            dist = np.full((n, n), inf)
            first = np.full((n, n), -1, dtype=int)
            np.fill_diagonal(dist, 0.0)
            for ai, arc in enumerate(arcs):
                _, _, u, v, cap, weight = arc
                if cap > 1e-9 and weight < dist[u, v]:
                    dist[u, v], first[u, v] = weight, ai
            for mid in range(n):
                for u in range(n):
                    if dist[u, mid] >= inf:
                        continue
                    cand = dist[u, mid] + dist[mid]
                    better = cand < dist[u] - 1e-9
                    dist[u, better] = cand[better]
                    first[u, better] = first[u, mid]

            sink_list = []
            for i, (node, kk) in enumerate(zip(sink_nodes, sink_ks)):
                if int(kk) == k and need.get((int(node), k), 0.0) > 1e-8:
                    pi = self.val(sink_pi[i]) if i < len(sink_pi) else 1.0
                    sink_list.append((int(node), need[(int(node), k)], pi))
            sink_list.sort(key=lambda x: -x[2])

            for sink, demand, _ in sink_list:
                remaining = demand
                origins = sorted(range(n), key=lambda x: dist[x, sink])
                for origin in origins:
                    if remaining <= 1e-8 or origin == sink or dist[origin, sink] >= inf:
                        continue
                    ai = int(first[origin, sink])
                    if ai < 0:
                        continue
                    s, route, u, v, arc_cap, _ = arcs[ai]
                    available = stock[u, k] + supply[u, k]
                    qty = min(remaining, available, arc_cap, edge_left[route[0]])
                    if qty <= 1e-8:
                        continue
                    flows[s] += qty
                    take_stock = min(qty, stock[u, k])
                    stock[u, k] -= take_stock
                    supply[u, k] = max(0.0, supply[u, k] - (qty - take_stock))
                    for re in route:
                        edge_left[re] = max(0.0, edge_left[re] - qty)
                    remaining -= qty

        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}
