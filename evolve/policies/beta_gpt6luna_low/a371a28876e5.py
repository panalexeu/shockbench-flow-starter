# -2.108768480536383
import heapq
import numpy as np


class Agent:
    def __init__(self, config=None):
        self.static = config["static"]
        self.layout = config["layout"]
        self.edges = self.static["edges"]
        self.slots = self.static["action_slots"]
        self.nn = len(self.static["nodes"]["id"])
        self.nk = len(self.static["commodities"]["id"])
        self.ns = len(self.slots["edge"])
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
        flows = np.zeros(self.ns, dtype=float)
        tails, heads = self.edges["tail"], self.edges["head"]
        lane_edges = self.static.get("lanes", {}).get("edges", [])
        slot_edges = self.slots["edge"]
        slot_ks = self.slots["k"]
        slot_lanes = self.slots.get("lane", [None] * self.ns)
        week = int(np.asarray(obs.get("week", [1])).ravel()[0])

        stock = np.zeros((self.nn, self.nk), dtype=float)
        sq = np.asarray(obs.get("stock.qty", [])).ravel()
        for i, pair in enumerate(self.layout.get("stock_slots", [])):
            if i < len(sq):
                n, k = map(int, pair)
                stock[n, k] += max(0.0, self.val(sq[i]))
        supply = np.zeros_like(stock)
        av = np.asarray(obs.get("graph_now.supply.avail", [])).ravel()
        for i, pair in enumerate(self.layout.get("supply_slots", [])):
            if i < len(av):
                n, k = map(int, pair)
                supply[n, k] += max(0.0, self.val(av[i]))

        sinks = self.static.get("sinks", {})
        snodes = [int(x) for x in sinks.get("node", [])]
        sks = [int(x) for x in sinks.get("k", [])]
        penalties = sinks.get("pi", [1.0] * len(snodes))
        backlog = np.asarray(obs.get("backlog.qty", [])).ravel()
        forecast = np.asarray(obs.get("demand_forecast.qty", []), dtype=float)
        # Cover immediate demand and a modest one-week safety buffer.
        need = {}
        for i, (n, k) in enumerate(zip(snodes, sks)):
            f = 0.0
            if forecast.ndim == 2 and i < forecast.shape[0]:
                f = sum(max(0.0, self.val(x)) for x in forecast[i, :min(2, forecast.shape[1])])
            b = max(0.0, self.val(backlog[i])) if i < len(backlog) else 0.0
            need[(n, k)] = max(0.0, b + f - stock[n, k])

        # Credit shipments already scheduled to reach their current edge head soon.
        pe = np.asarray(obs.get("pipeline.edge", [])).ravel()
        pk = np.asarray(obs.get("pipeline.k", [])).ravel()
        pq = np.asarray(obs.get("pipeline.qty", [])).ravel()
        pa = np.asarray(obs.get("pipeline.arrival_week", [])).ravel()
        live = np.asarray(obs.get("pipeline.qty.observed", np.ones(len(pq)))).ravel()
        for i in range(min(len(pe), len(pk), len(pq), len(pa))):
            if i < len(live) and not live[i]:
                continue
            e, k = int(pe[i]), int(pk[i])
            if 0 <= e < len(heads) and 0 <= k < self.nk and int(pa[i]) <= week + 2:
                key = (int(heads[e]), k)
                if key in need:
                    need[key] = max(0.0, need[key] - max(0.0, self.val(pq[i])))

        cap_now = np.asarray(obs.get("graph_now.u", [])).ravel()
        cost_now = np.asarray(obs.get("graph_now.c", [])).ravel()
        tau_now = np.asarray(obs.get("graph_now.tau", [])).ravel()
        mask = np.asarray(obs.get("action_mask", np.ones(self.ns))).ravel()
        mask_seen = bool(np.asarray(obs.get("action_mask.observed", [1])).ravel()[0])
        prohibited = np.asarray(obs.get("slot_mask", np.zeros(self.ns))).ravel()
        edge_left = np.full(len(tails), np.inf)
        for e in range(len(tails)):
            if e < len(cap_now) and np.isfinite(cap_now[e]):
                edge_left[e] = max(0.0, float(cap_now[e]))

        for k in range(self.nk):
            # Build a directed graph of usable action slots and find cheapest
            # paths to each sink. Each slot is a single dispatch decision.
            arcs = []
            adj = [[] for _ in range(self.nn)]
            for s in range(self.ns):
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
                    if re < len(cost_now):
                        cost += max(0.0, self.val(cost_now[re]))
                    elif re < len(self.edges.get("c0", [])):
                        cost += max(0.0, self.val(self.edges["c0"][re]))
                    tau += max(0.0, self.val(tau_now[re], 1.0)) if re < len(tau_now) else 1.0
                cap = min(cap, *(edge_left[re] for re in route))
                if cap <= 1e-9:
                    continue
                u, v = int(tails[route[0]]), int(heads[route[-1]])
                ai = len(arcs)
                arcs.append((s, route, u, v, cap, cost + 0.15 * tau))
                adj[u].append((v, ai))

            sink_work = []
            for i, (n, kk) in enumerate(zip(snodes, sks)):
                d = need.get((n, kk), 0.0)
                if kk == k and d > 1e-8:
                    pi = max(0.0, self.val(penalties[i] if i < len(penalties) else 1.0, 1.0))
                    sink_work.append([n, d, pi])
            sink_work.sort(key=lambda x: -x[2])

            # Allocate each sink from its least-cost currently stocked origin.
            for sink, remaining, _ in sink_work:
                while remaining > 1e-8:
                    best = None
                    for origin in range(self.nn):
                        available = stock[origin, k] + supply[origin, k]
                        if origin == sink or available <= 1e-8:
                            continue
                        dist = [float("inf")] * self.nn
                        first = [-1] * self.nn
                        dist[origin] = 0.0
                        heap = [(0.0, origin)]
                        while heap:
                            du, u = heapq.heappop(heap)
                            if du != dist[u]:
                                continue
                            for v, ai in adj[u]:
                                a = arcs[ai]
                                if a[4] <= 1e-9:
                                    continue
                                nd = du + a[5]
                                if nd < dist[v]:
                                    dist[v] = nd
                                    first[v] = ai if u == origin else first[u]
                                    heapq.heappush(heap, (nd, v))
                        if dist[sink] < float("inf") and first[sink] >= 0:
                            ai = first[sink]
                            a = arcs[ai]
                            if best is None or dist[sink] < best[0]:
                                best = (dist[sink], ai, available)
                    if best is None:
                        break
                    _, ai, available = best
                    s, route, origin, _, arc_cap, _ = arcs[ai]
                    qty = min(remaining, available, arc_cap, *(edge_left[re] for re in route))
                    if qty <= 1e-8:
                        break
                    flows[s] += qty
                    use_stock = min(qty, stock[origin, k])
                    stock[origin, k] -= use_stock
                    supply[origin, k] = max(0.0, supply[origin, k] - (qty - use_stock))
                    for re in route:
                        edge_left[re] = max(0.0, edge_left[re] - qty)
                    arcs[ai] = (s, route, origin, arcs[ai][3], max(0.0, arc_cap - qty), arcs[ai][5])
                    remaining -= qty

        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}