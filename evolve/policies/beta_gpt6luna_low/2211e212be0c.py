# -2.110498143695876
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
        self.override_qty = np.zeros(config["spaces"]["action"]["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(config["spaces"]["action"]["release_mode"]["shape"], dtype=np.int64)

    @staticmethod
    def val(x):
        try:
            x = float(x)
            return max(0.0, x) if np.isfinite(x) else 0.0
        except Exception:
            return 0.0

    def act(self, obs):
        n, nk, ns = self.nodes, self.nk, self.nslot
        flows = np.zeros(ns, dtype=float)
        week = int(np.asarray(obs.get("week", [1])).ravel()[0])
        tails, heads = self.edges["tail"], self.edges["head"]
        lane_edges = self.static.get("lanes", {}).get("edges", [])
        slot_edges = self.slots["edge"]
        slot_ks = self.slots["k"]
        slot_lanes = self.slots.get("lane", [None] * ns)

        stock = np.zeros((n, nk))
        for i, (node, k) in enumerate(self.layout.get("stock_slots", [])):
            q = np.asarray(obs.get("stock.qty", [])).ravel()
            if i < len(q):
                stock[int(node), int(k)] += self.val(q[i])
        supply = np.zeros((n, nk))
        avail = np.asarray(obs.get("graph_now.supply.avail", [])).ravel()
        for i, (node, k) in enumerate(self.layout.get("supply_slots", [])):
            if i < len(avail):
                supply[int(node), int(k)] += self.val(avail[i])

        forecast = np.asarray(obs.get("demand_forecast.qty", []), dtype=float)
        backlog = np.asarray(obs.get("backlog.qty", [])).ravel()
        sink_nodes = self.static["sinks"]["node"]
        sink_ks = self.static["sinks"]["k"]
        sink_pi = self.static["sinks"].get("pi", [1.0] * len(sink_nodes))
        demand = {}
        for i, (node, k) in enumerate(zip(sink_nodes, sink_ks)):
            # Cover current/near-term demand without building a long-horizon stockpile.
            f = sum(self.val(x) for x in forecast[i, :min(2, forecast.shape[1])]) if forecast.ndim == 2 and i < forecast.shape[0] else 0.0
            b = self.val(backlog[i]) if i < len(backlog) else 0.0
            demand[(int(node), int(k))] = b + f

        cap_now = np.asarray(obs.get("graph_now.u", [])).ravel()
        cost_now = np.asarray(obs.get("graph_now.c", [])).ravel()
        tau_now = np.asarray(obs.get("graph_now.tau", [])).ravel()
        mask = np.asarray(obs.get("action_mask", np.ones(ns))).ravel()
        mask_seen = bool(np.asarray(obs.get("action_mask.observed", [1])).ravel()[0])
        prohibited = np.asarray(obs.get("slot_mask", np.zeros(ns))).ravel()
        cp_nodes = [int(x) for x in self.layout.get("chokepoints", [])]
        cp_pos = {node: i for i, node in enumerate(cp_nodes)}
        opened = np.asarray(obs.get("graph_now.open", [])).ravel()
        edge_remaining = np.full(len(tails), np.inf)
        for e in range(len(tails)):
            if e < len(cap_now) and np.isfinite(cap_now[e]):
                edge_remaining[e] = max(0.0, float(cap_now[e]))

        for k in range(nk):
            arcs = []
            for s in range(ns):
                if int(slot_ks[s]) != k:
                    continue
                if mask_seen and s < len(mask) and mask[s] < 0.5:
                    continue
                if not mask_seen and s < len(prohibited) and prohibited[s] > 0.5:
                    continue
                e = int(slot_edges[s])
                lane = slot_lanes[s]
                route = [e]
                if lane is not None:
                    try:
                        li = int(lane)
                        if 0 <= li < len(lane_edges) and lane_edges[li]:
                            route = [int(x) for x in lane_edges[li]]
                    except Exception:
                        pass
                if not route:
                    continue
                route_cap, cost, tau, open_factor = np.inf, 0.0, 0.0, 1.0
                for re in route:
                    if re < len(cap_now) and np.isfinite(cap_now[re]):
                        route_cap = min(route_cap, max(0.0, float(cap_now[re])))
                    cost += self.val(cost_now[re]) if re < len(cost_now) else self.val(self.edges.get("c0", [0] * len(tails))[re])
                    tau += self.val(tau_now[re]) if re < len(tau_now) else 1.0
                    cp = cp_pos.get(int(heads[re]))
                    if cp is not None and cp < len(opened):
                        open_factor *= min(1.0, self.val(opened[cp]))
                route_cap = min(route_cap, edge_remaining[route[0]])
                if route_cap <= 0 or open_factor <= 0:
                    continue
                u, v = int(tails[route[0]]), int(heads[route[-1]])
                # Freight dominates only modestly: prefer quick routes when costs are close.
                weight = cost + 0.15 * tau
                arcs.append((s, route, u, v, route_cap * open_factor, weight, tau))

            # Shortest paths over this commodity's currently usable action arcs.
            inf = 1e15
            dist = np.full((n, n), inf)
            first = np.full((n, n), -1, dtype=int)
            for x in range(n):
                dist[x, x] = 0.0
            for ai, arc in enumerate(arcs):
                _, _, u, v, cap, weight, _ = arc
                if cap > 0 and weight < dist[u, v]:
                    dist[u, v], first[u, v] = weight, ai
            for mid in range(n):
                for u in range(n):
                    if dist[u, mid] >= inf:
                        continue
                    cand = dist[u, mid] + dist[mid]
                    better = cand < dist[u] - 1e-9
                    dist[u, better] = cand[better]
                    first[u, better] = first[u, mid]

            sinks = [(node, need, self.val(sink_pi[i])) for i, ((node, kk), need) in enumerate(demand.items()) if kk == k and need > 1e-8]
            sinks.sort(key=lambda x: -x[2])
            for sink, need, _ in sinks:
                need = max(0.0, need - stock[sink, k])
                while need > 1e-8:
                    origins = sorted(range(n), key=lambda x: dist[x, sink])
                    chosen = None
                    for origin in origins:
                        if origin == sink or dist[origin, sink] >= inf:
                            continue
                        ai = int(first[origin, sink])
                        if ai >= 0 and stock[origin, k] + supply[origin, k] > 1e-8:
                            chosen = (origin, ai)
                            break
                    if chosen is None:
                        break
                    origin, ai = chosen
                    s, route, u, v, cap, _, _ = arcs[ai]
                    qty = min(need, stock[u, k] + supply[u, k], cap)
                    if qty <= 1e-8:
                        break
                    flows[s] += qty
                    take = min(qty, stock[u, k])
                    stock[u, k] -= take
                    supply[u, k] -= qty - take
                    for re in route:
                        edge_remaining[re] = max(0.0, edge_remaining[re] - qty)
                    need -= qty
        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}
