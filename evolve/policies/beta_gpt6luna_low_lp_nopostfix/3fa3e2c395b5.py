# -2.1072837114335243
import numpy as np
from scipy.optimize import linprog


class Agent:
    def __init__(self, config):
        self.S = config['static']
        self.L = config['layout']
        self.E = self.S['edges']
        self.slots = self.S['action_slots']
        self.lanes = self.S.get('lanes', {})
        self.nn = len(self.S.get('nodes', {}).get('id', []))
        self.ne = len(self.E.get('tail', []))
        self.nk = len(self.S.get('commodities', {}).get('id', []))
        self.ns = len(self.slots.get('edge', []))
        self.nd = len(self.L.get('demands', []))

    @staticmethod
    def _arr(obs, key, default):
        try:
            return np.asarray(obs.get(key, default))
        except Exception:
            return np.asarray(default)

    @staticmethod
    def _get(a, i, default=0.0):
        try:
            x = a[i]
            if x is None:
                return default
            x = float(x)
            return x if np.isfinite(x) else default
        except (IndexError, TypeError, ValueError):
            return default

    def act(self, obs):
        ns, nd, nn, nk, ne = self.ns, self.nd, self.nn, self.nk, self.ne
        if not ns or not nd or not nn or not nk:
            return {'flows': np.zeros(ns, dtype=float)}
        S, L, E = self.S, self.L, self.E
        week = int(self._arr(obs, 'week', [1]).reshape(-1)[0])
        T = int(S.get('T', S.get('instance', {}).get('T', 52)))
        forecast = self._arr(obs, 'demand_forecast.qty', np.zeros((nd, 1)))
        if forecast.ndim != 2:
            forecast = np.zeros((nd, 1))
        H = min(8, forecast.shape[1], max(1, T - week + 1))
        if H <= 0:
            return {'flows': np.zeros(ns, dtype=float)}

        u = self._arr(obs, 'graph_now.u', E.get('u0', [])).reshape(-1)
        c = self._arr(obs, 'graph_now.c', E.get('c0', [])).reshape(-1)
        tau = self._arr(obs, 'graph_now.tau', E.get('tau0', [])).reshape(-1)
        tariff = self._arr(obs, 'graph_now.tariff', np.zeros((ne, nk)))
        prohibited = self._arr(obs, 'graph_now.prohibited', np.zeros((ne, nk)))
        mask = self._arr(obs, 'action_mask', np.ones(ns)).reshape(-1)
        slot_e = self.slots['edge']
        slot_k = self.slots['k']
        slot_lane = self.slots.get('lane', [None] * ns)
        values = S.get('commodities', {}).get('v', [])
        edge_K = E.get('K', [])
        paths, tails, heads, delays, route_cost, route_ub = [], [], [], [], [], []
        for s in range(ns):
            e, k = int(slot_e[s]), int(slot_k[s])
            lane = slot_lane[s] if s < len(slot_lane) else None
            path = [e]
            if lane is not None:
                try:
                    path = [int(x) for x in self.lanes['edges'][int(lane)]]
                except (KeyError, IndexError, TypeError, ValueError):
                    path = []
            valid = bool(path) and 0 <= k < nk and all(0 <= x < ne for x in path)
            if valid:
                for x in path:
                    if x < len(edge_K) and edge_K[x] and k not in edge_K[x]:
                        valid = False
                    if x < prohibited.shape[0] and k < prohibited.shape[1] and prohibited[x, k] > 0:
                        valid = False
                if s < len(mask) and mask[s] <= 0:
                    valid = False
            cost, delay, ub = 0.0, 0, 1e12
            if valid:
                for x in path:
                    cap = self._get(u, x, self._get(E.get('u0', []), x, 1e12))
                    if np.isfinite(cap):
                        ub = min(ub, max(0.0, cap))
                    cost += max(0.0, self._get(c, x, self._get(E.get('c0', []), x)))
                    try:
                        if x < tariff.shape[0] and k < tariff.shape[1]:
                            cost += max(0.0, float(tariff[x, k])) * max(0.0, self._get(values, k))
                    except (IndexError, TypeError, ValueError):
                        pass
                    delay += max(0, int(self._get(tau, x, self._get(E.get('tau0', []), x))))
                tail, head = int(E['tail'][path[0]]), int(E['head'][path[-1]])
            else:
                path, tail, head, ub = [], 0, 0, 0.0
            paths.append(path)
            tails.append(tail)
            heads.append(head)
            delays.append(max(1, delay))
            route_cost.append(cost)
            route_ub.append(ub)

        # Variables are dispatches by week/slot, then service by week/demand.
        nx, ny = H * ns, H * nd
        nv = nx + ny
        xid = lambda h, s: h * ns + s
        yid = lambda h, d: nx + h * nd + d
        obj = np.zeros(nv)
        bounds = [(0.0, route_ub[s]) for h in range(H) for s in range(ns)]
        demand = np.zeros((H, nd))
        backlog = self._arr(obs, 'backlog.qty', np.zeros(nd)).reshape(-1)
        penalties = S.get('sinks', {}).get('pi', [])
        for h in range(H):
            for d in range(nd):
                f = max(0.0, self._get(forecast[d] if d < forecast.shape[0] else [], h))
                demand[h, d] = f + (max(0.0, self._get(backlog, d)) if h == 0 else 0.0)
                bounds.append((0.0, demand[h, d]))
                obj[yid(h, d)] = -max(0.0, self._get(penalties, d, 1.0)) * (0.995 ** h)
            for s in range(ns):
                obj[xid(h, s)] = route_cost[s] * (0.995 ** h)

        available = np.zeros((nn, nk))
        stock = self._arr(obs, 'stock.qty', []).reshape(-1)
        for i, pair in enumerate(L.get('stock_slots', [])):
            if len(pair) >= 2 and i < len(stock):
                n, k = int(pair[0]), int(pair[1])
                if 0 <= n < nn and 0 <= k < nk:
                    available[n, k] += max(0.0, self._get(stock, i))
        supply = self._arr(obs, 'graph_now.supply.avail', []).reshape(-1)
        for i, pair in enumerate(L.get('supply_slots', [])):
            if len(pair) >= 2 and i < len(supply):
                n, k = int(pair[0]), int(pair[1])
                if 0 <= n < nn and 0 <= k < nk:
                    available[n, k] += max(0.0, self._get(supply, i))

        # Exogenous in-transit cargo is available at its destination in its arrival week.
        arrivals = np.zeros((H, nn, nk))
        pe = self._arr(obs, 'pipeline.edge', []).reshape(-1)
        pk = self._arr(obs, 'pipeline.k', []).reshape(-1)
        pq = self._arr(obs, 'pipeline.qty', []).reshape(-1)
        pa = self._arr(obs, 'pipeline.arrival_week', []).reshape(-1)
        for i in range(min(len(pe), len(pk), len(pq), len(pa))):
            e, k = int(pe[i]), int(pk[i])
            h = int(pa[i]) - week
            if 0 <= h < H and 0 <= e < ne and 0 <= k < nk:
                n = int(E['head'][e])
                if 0 <= n < nn:
                    arrivals[h, n, k] += max(0.0, self._get(pq, i))

        A, b = [], []
        sink_node = S.get('sinks', {}).get('node', [])
        sink_k = S.get('sinks', {}).get('k', [])
        # Cumulative balances at every time boundary permit inventory carryover.
        for h in range(H):
            for n in range(nn):
                for k in range(nk):
                    row = np.zeros(nv)
                    for q in range(h + 1):
                        for s in range(ns):
                            if int(slot_k[s]) != k:
                                continue
                            if tails[s] == n:
                                row[xid(q, s)] += 1.0
                            arrival_h = q + delays[s]
                            if arrival_h <= h and heads[s] == n:
                                row[xid(q, s)] -= 1.0
                        for d in range(nd):
                            if d < len(sink_node) and d < len(sink_k) and int(sink_node[d]) == n and int(sink_k[d]) == k:
                                row[yid(q, d)] += 1.0
                    rhs = available[n, k] + float(np.sum(arrivals[:h + 1, n, k]))
                    if np.any(row):
                        A.append(row)
                        b.append(rhs)

        # A route consumes capacity on every edge it traverses in its dispatch week.
        for h in range(H):
            for e in range(ne):
                cap = max(0.0, self._get(u, e, self._get(E.get('u0', []), e, 1e12)))
                row = np.zeros(nv)
                for s in range(ns):
                    if e in paths[s]:
                        row[xid(h, s)] += 1.0
                if np.any(row):
                    A.append(row)
                    b.append(cap)

        # Pool throughput at chokepoints, separated by commodity pool.
        cp_nodes = list(L.get('chokepoints', []))
        cp_row = {int(n): j for j, n in enumerate(cp_nodes)}
        op = self._arr(obs, 'graph_now.open', np.ones(len(cp_nodes))).reshape(-1)
        ktb = self._arr(obs, 'graph_now.kappa.tb', np.full(len(cp_nodes), 1e12)).reshape(-1)
        kct = self._arr(obs, 'graph_now.kappa.ct', np.full(len(cp_nodes), 1e12)).reshape(-1)
        pools = S.get('commodities', {}).get('pool', ['tb'] * nk)
        for h in range(H):
            for j, node in enumerate(cp_nodes):
                for pool, kap in (('tb', ktb), ('ct', kct)):
                    cap = max(0.0, self._get(op, j, 1.0)) * max(0.0, self._get(kap, j, 1e12))
                    row = np.zeros(nv)
                    for s in range(ns):
                        k = int(slot_k[s])
                        if k < len(pools) and pools[k] == pool and any(int(E['head'][e]) == int(node) for e in paths[s]):
                            row[xid(h, s)] += 1.0
                    if np.any(row):
                        A.append(row)
                        b.append(cap)

        try:
            res = linprog(obj, A_ub=np.asarray(A), b_ub=np.asarray(b), bounds=bounds, method='highs')
            flows = np.maximum(0.0, res.x[:ns]) if res.success else np.zeros(ns)
        except Exception:
            flows = np.zeros(ns)
        return {'flows': flows.astype(float)}