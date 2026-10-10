# 0.6085758013649495
import numpy as np


class Agent:
    def __init__(self, config=None):
        self.config = config or {}
        self.static = self.config.get("static", {})
        self.layout = self.config.get("layout", {})
        self.T = int(self.config.get("T", self.static.get("T", 52)))
        slots = self.static.get("action_slots", {})
        edges = self.static.get("edges", {})
        self.slot_edge = np.asarray(slots.get("edge", []), dtype=int)
        self.slot_k = np.asarray(slots.get("k", []), dtype=int)
        self.slot_lane = list(slots.get("lane", [None] * len(self.slot_edge)))
        self.n = len(self.slot_edge)
        self.edge_tail = np.asarray(edges.get("tail", []), dtype=int)
        self.edge_head = np.asarray(edges.get("head", []), dtype=int)
        self.edge_u0 = np.asarray([0.0 if x is None else float(x) for x in edges.get("u0", [])], dtype=float)
        self.edge_tau0 = np.asarray([1.0 if x is None else float(x) for x in edges.get("tau0", [])], dtype=float)
        self.edge_c0 = np.asarray([0.0 if x is None else float(x) for x in edges.get("c0", [])], dtype=float)
        self.lane_edges = self.static.get("lanes", {}).get("edges", [])
        self.lane_chokes = self.static.get("lanes", {}).get("chokepoints", [])
        self.nodes = list(self.static.get("nodes", {}).get("id", []))
        self.commodities = list(self.static.get("commodities", {}).get("id", []))
        self.node_ix = {str(x): i for i, x in enumerate(self.nodes)}
        self.comm_ix = {str(x): i for i, x in enumerate(self.commodities)}
        self.cp_ix = {}
        for j, name in enumerate(self.layout.get("chokepoints", [])):
            if str(name) in self.node_ix:
                self.cp_ix[self.node_ix[str(name)]] = j

        self.paths = []
        for e, lane in zip(self.slot_edge, self.slot_lane):
            try:
                self.paths.append([int(x) for x in self.lane_edges[int(lane)]] if lane is not None else [int(e)])
            except (TypeError, ValueError, IndexError):
                self.paths.append([int(e)])

        self.stock_index = {}
        for j, item in enumerate(self.layout.get("stock_slots", [])):
            try:
                n, k = item[0], item[1]
                ni = self.node_ix.get(str(n), int(n) if isinstance(n, (int, np.integer)) else -1)
                ki = self.comm_ix.get(str(k), int(k) if isinstance(k, (int, np.integer)) else -1)
                if ni >= 0 and ki >= 0:
                    self.stock_index[(int(ni), int(ki))] = j
            except (TypeError, ValueError, IndexError):
                pass
        self.supply_index = {}
        for j, item in enumerate(self.layout.get("supply_slots", [])):
            try:
                n, k = item[0], item[1]
                ni = self.node_ix.get(str(n), int(n) if isinstance(n, (int, np.integer)) else -1)
                ki = self.comm_ix.get(str(k), int(k) if isinstance(k, (int, np.integer)) else -1)
                if ni >= 0 and ki >= 0:
                    self.supply_index[(int(ni), int(ki))] = j
            except (TypeError, ValueError, IndexError):
                pass

        sinks = self.static.get("sinks", {})
        self.sink_node = np.asarray(sinks.get("node", []), dtype=int)
        self.sink_k = np.asarray(sinks.get("k", []), dtype=int)
        self.sink_backlog = np.asarray(sinks.get("backlog", [False] * len(self.sink_node)), dtype=bool)
        self.sink_penalty = np.asarray(sinks.get("pi", np.ones(len(self.sink_node))), dtype=float)
        self.demands_by_key = {}
        for d, (n, k) in enumerate(zip(self.sink_node, self.sink_k)):
            self.demands_by_key[(int(n), int(k))] = d

        self.outgoing = {}
        self.groups = {}
        for i, path in enumerate(self.paths):
            if not path:
                continue
            e0 = int(path[0])
            if not 0 <= e0 < len(self.edge_tail):
                continue
            key = (int(self.edge_tail[e0]), int(self.slot_k[i]))
            self.groups.setdefault(key, []).append(i)
            self.outgoing.setdefault(key[0], []).append(i)

        self.production = []
        inst = self.static.get("instance", {})
        params = inst.get("params", {}) if isinstance(inst, dict) else {}
        for key in ("fabs", "osats", "production", "recipes"):
            val = params.get(key) if isinstance(params, dict) else None
            if isinstance(val, (list, dict)):
                self.production.append(val)

    def _stock_value(self, key, stock, stock_obs, supply):
        si = self.stock_index.get(key)
        if si is not None and si < len(stock) and si < len(stock_obs) and stock_obs[si]:
            return max(0.0, float(stock[si]))
        qi = self.supply_index.get(key)
        if qi is not None and qi < len(supply):
            return max(0.0, float(supply[qi]))
        return 0.0

    def _route_ok(self, i, mask, cap, opened, tau, week):
        if i >= len(mask) or mask[i] <= 0:
            return False
        path = self.paths[i]
        if not path:
            return False
        total_tau = 0.0
        for e in path:
            if e < 0 or e >= len(cap) or cap[e] <= 1e-9:
                return False
            total_tau += max(0.0, float(tau[e])) if e < len(tau) else 1.0
        if total_tau > self.T - week + 1:
            return False
        lane = self.slot_lane[i]
        if lane is not None:
            try:
                cps = self.lane_chokes[int(lane)]
            except (TypeError, ValueError, IndexError):
                cps = []
            for node in cps:
                cp = self.cp_ix.get(int(node))
                if cp is not None and cp < len(opened) and opened[cp] <= 1e-6:
                    return False
        return True

    def _read_pipeline(self, o, week, horizon=12):
        arrivals = {}
        fields = ["pipeline.edge", "pipeline.k", "pipeline.qty", "pipeline.arrival_week"]
        arrays = [np.asarray(o.get(x, [])) for x in fields]
        if any(a.size == 0 for a in arrays):
            return arrivals
        pe, pk, pq, pa = [a.reshape(-1) for a in arrays]
        live = np.asarray(o.get("pipeline.qty.observed", np.ones(len(pq))), dtype=bool).reshape(-1)
        for j in range(min(map(len, (pe, pk, pq, pa, live)))):
            if not live[j] or pq[j] <= 0:
                continue
            e = int(pe[j])
            if not 0 <= e < len(self.edge_head):
                continue
            remaining = int(pa[j]) - week
            if remaining < 0:
                remaining = 0
            key = (int(self.edge_head[e]), int(pk[j]))
            arrivals.setdefault(key, np.zeros(horizon + 1, dtype=float))
            arrivals[key][min(horizon, remaining)] += float(pq[j])
        return arrivals

    def _demand_rates(self, o):
        forecast = np.asarray(o.get("demand_forecast.qty", np.zeros((len(self.sink_node), 1))), dtype=float)
        obs = np.asarray(o.get("demand_forecast.qty.observed", np.ones_like(forecast)), dtype=bool)
        backlog = np.asarray(o.get("backlog.qty", np.zeros(len(self.sink_node))), dtype=float).reshape(-1)
        rates = np.zeros(len(self.sink_node), dtype=float)
        for d in range(len(rates)):
            if forecast.ndim == 2 and d < forecast.shape[0]:
                vals = forecast[d]
                ok = obs[d] if obs.ndim == 2 and d < obs.shape[0] else np.ones(len(vals), dtype=bool)
                vals = vals[ok]
                if len(vals):
                    # Smooth the weekly forecast and discount the distant tail.
                    w = np.exp(-np.arange(len(vals)) / 5.0)
                    rates[d] = float(np.sum(vals * w) / max(1e-9, w.sum()))
            if d < len(backlog):
                rates[d] += max(0.0, float(backlog[d])) / 5.0
        return np.maximum(rates, 0.0)

    def act(self, observation):
        o = observation
        flows = np.zeros(self.n, dtype=float)
        if not self.n:
            return {"flows": flows}
        week = int(np.asarray(o.get("week", [1])).reshape(-1)[0])
        mask = np.asarray(o.get("action_mask", np.ones(self.n)), dtype=float).reshape(-1)
        if len(mask) != self.n:
            mask = np.ones(self.n, dtype=float)
        cap = np.asarray(o.get("graph_now.u", self.edge_u0), dtype=float).reshape(-1)
        tau = np.asarray(o.get("graph_now.tau", self.edge_tau0), dtype=float).reshape(-1)
        cost = np.asarray(o.get("graph_now.c", self.edge_c0), dtype=float).reshape(-1)
        opened = np.asarray(o.get("graph_now.open", []), dtype=float).reshape(-1)
        stock = np.asarray(o.get("stock.qty", []), dtype=float).reshape(-1)
        stock_obs = np.asarray(o.get("stock.qty.observed", np.ones(len(stock))), dtype=bool).reshape(-1)
        supply = np.asarray(o.get("graph_now.supply.avail", []), dtype=float).reshape(-1)
        forecast = np.asarray(o.get("demand_forecast.qty", np.zeros((len(self.sink_node), 1))), dtype=float)
        backlog = np.asarray(o.get("backlog.qty", np.zeros(len(self.sink_node))), dtype=float).reshape(-1)
        arrivals = self._read_pipeline(o, week, 12)
        rates = self._demand_rates(o)

        # Target stock at sinks and intermediate nodes is expressed in units of
        # expected weekly throughput; deeper nodes receive the aggregate needs of
        # all downstream sink paths in the static network.
        need = {}
        for d, (node, k) in enumerate(zip(self.sink_node, self.sink_k)):
            rate = rates[d] if d < len(rates) else 0.0
            horizon = 2.8
            target = rate * horizon + (max(0.0, backlog[d]) if d < len(backlog) else 0.0)
            key = (int(node), int(k))
            onhand = self._stock_value(key, stock, stock_obs, supply)
            arriving = sum(float(x.sum()) for x in [arrivals.get(key, np.zeros(1))])
            need[key] = max(0.0, target - onhand - arriving)

        # Work backward from sink requirements through action-slot routes. Each
        # requested shipment is capped by observed source availability and route
        # capacity, and alternate routes share the same source/edge constraints.
        requested = np.zeros(self.n, dtype=float)
        remaining_need = dict(need)
        ordered = sorted(range(self.n), key=lambda i: sum(max(0.0, float(tau[e])) for e in self.paths[i] if e < len(tau)))
        # Demand propagation over several passes lets intermediate commodities
        # acquire replenishment targets even when they are multiple stages upstream.
        for _ in range(3):
            for i in reversed(ordered):
                if not self._route_ok(i, mask, cap, opened, tau, week):
                    continue
                path = self.paths[i]
                dest = (int(self.edge_head[path[-1]]), int(self.slot_k[i]))
                if dest not in remaining_need:
                    continue
                qty_need = max(0.0, remaining_need[dest])
                if qty_need <= 1e-8:
                    continue
                src = (int(self.edge_tail[path[0]]), int(self.slot_k[i]))
                avail = self._stock_value(src, stock, stock_obs, supply)
                already = sum(requested[j] for j in self.groups.get(src, []))
                avail = max(0.0, avail - already)
                if avail <= 1e-8:
                    continue
                route_cap = min(max(0.0, float(cap[e])) for e in path)
                if route_cap <= 0:
                    continue
                route_cost = sum(max(0.0, float(cost[e])) if e < len(cost) else 0.0 for e in path)
                # Prefer economical routes but don't avoid the only useful way to serve need.
                qty = min(qty_need, avail, route_cap)
                if qty <= 0:
                    continue
                requested[i] += qty
                remaining_need[dest] -= qty
                # For internal production nodes, reserve enough incoming material
                # to keep their outgoing routes near the observed downstream need.
                if dest not in self.demands_by_key:
                    src_need = remaining_need.get(dest, 0.0)
                    if src_need > 0:
                        remaining_need[dest] = src_need

        # For terminal and production inputs not captured by sink-side backprop,
        # preserve part of the normal feasible traffic while avoiding duplicate
        # dispatches beyond local inventory and downstream demand.
        for key, ids in self.groups.items():
            available = self._stock_value(key, stock, stock_obs, supply)
            used = sum(requested[i] for i in ids)
            room = max(0.0, available - used)
            if room <= 1e-8:
                continue
            candidates = [i for i in ids if self._route_ok(i, mask, cap, opened, tau, week)]
            if not candidates:
                continue
            # Route value: short lead time and low cost; nominal capacity gives a
            # conservative fallback for source/input flows.
            scores = []
            for i in candidates:
                path = self.paths[i]
                rc = sum(max(0.0, float(cost[e])) if e < len(cost) else 0.0 for e in path)
                rt = sum(max(0.0, float(tau[e])) if e < len(tau) else 1.0 for e in path)
                scores.append(1.0 / ((1.0 + rc) ** 0.45 * (1.0 + 0.12 * rt)))
            scores = np.asarray(scores, dtype=float)
            scores /= max(1e-12, float(scores.sum()))
            for j, i in enumerate(candidates):
                if requested[i] > 0:
                    continue
                path = self.paths[i]
                limit = min(max(0.0, float(cap[e])) for e in path)
                qty = min(room * scores[j], limit)
                requested[i] = qty

        # Enforce each action edge's shared capacity and each source inventory.
        flows[:] = np.maximum(0.0, requested)
        for e in np.unique(self.slot_edge):
            e = int(e)
            if not 0 <= e < len(cap):
                continue
            ids = np.flatnonzero(self.slot_edge == e)
            total = float(flows[ids].sum())
            limit = max(0.0, float(cap[e]))
            if total > limit and total > 0:
                flows[ids] *= limit / total
        for key, ids0 in self.groups.items():
            ids = np.asarray(ids0, dtype=int)
            available = self._stock_value(key, stock, stock_obs, supply)
            total = float(flows[ids].sum())
            if total > available and total > 0:
                flows[ids] *= available / total
        flows *= (mask > 0)
        flows[~np.isfinite(flows)] = 0.0
        flows = np.maximum(flows, 0.0)
        return {
            "flows": flows,
            "override_qty": np.zeros(36, dtype=float),
            "release_mode": np.zeros(14, dtype=np.int64),
        }