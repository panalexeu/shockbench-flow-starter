# -2.1073502733969542
import numpy as np
from scipy.optimize import linprog


class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.e = self.s['edges']
        self.slots = self.s['action_slots']
        self.lanes = self.s.get('lanes', {})
        self.ne = len(self.e.get('tail', []))
        self.nn = len(self.s.get('nodes', {}).get('id', []))
        self.nk = len(self.s.get('commodities', {}).get('id', []))
        self.ns = len(self.slots.get('edge', []))

    @staticmethod
    def arr(obs, key, fallback):
        try:
            return np.asarray(obs.get(key, fallback))
        except Exception:
            return np.asarray(fallback)

    @staticmethod
    def val(a, i, default=0.0):
        try:
            x = float(a[i])
            return x if np.isfinite(x) else default
        except (IndexError, TypeError, ValueError):
            return default

    def act(self, obs):
        ns, nd, nn, nk, ne = self.ns, len(self.l.get('demands', [])), self.nn, self.nk, self.ne
        zero = np.zeros(ns, dtype=float)
        if not ns or not nd or not nn or not nk:
            return {'flows': zero}
        S, L, E = self.s, self.l, self.e
        week = int(self.arr(obs, 'week', [1]).reshape(-1)[0])
        T = int(S.get('T', S.get('instance', {}).get('T', 52)))
        fc = self.arr(obs, 'demand_forecast.qty', np.zeros((nd, 1)))
        if fc.ndim != 2:
            fc = np.zeros((nd, 1))
        H = max(1, min(8, fc.shape[1], T - week + 1))

        u = self.arr(obs, 'graph_now.u', E.get('u0', [])).reshape(-1)
        c = self.arr(obs, 'graph_now.c', E.get('c0', [])).reshape(-1)
        tau = self.arr(obs, 'graph_now.tau', E.get('tau0', [])).reshape(-1)
        tariff = self.arr(obs, 'graph_now.tariff', np.zeros((ne, nk)))
        prohibited = self.arr(obs, 'graph_now.prohibited', np.zeros((ne, nk)))
        mask = self.arr(obs, 'action_mask', np.ones(ns)).reshape(-1)
        edge_K = E.get('K', [])
        values = S.get('commodities', {}).get('v', [])
        slot_e, slot_k = self.slots['edge'], self.slots['k']
        slot_lane = self.slots.get('lane', [None] * ns)

        paths, tails, heads, delays, costs, caps = [], [], [], [], [], []
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
            cap, cost, delay = 1e12, 0.0, 0
            if valid:
                for x in path:
                    cap = min(cap, max(0.0, self.val(u, x, self.val(E.get('u0', []), x, 1e12))))
                    cost += max(0.0, self.val(c, x, self.val(E.get('c0', []), x)))
                    if x < tariff.shape[0] and k < tariff.shape[1]:
                        cost += max(0.0, self.val(tariff[x], k)) * max(0.0, self.val(values, k))
                    delay += max(0, int(self.val(tau, x, self.val(E.get('tau0', []), x))))
                tails.append(int(E['tail'][path[0]]))
                heads.append(int(E['head'][path[-1]]))
            else:
                path, cap = [], 0.0
                tails.append(-1)
                heads.append(-1)
            paths.append(path)
            delays.append(max(1, delay))
            costs.append(cost)
            caps.append(cap)

        nx, nv = H * ns, H * ns + H * nd
        xid = lambda h, s: h * ns + s
        yid = lambda h, d: nx + h * nd + d
        obj = np.zeros(nv)
        bounds = [(0.0, caps[s]) for h in range(H) for s in range(ns)]
        backlog = self.arr(obs, 'backlog.qty', np.zeros(nd)).reshape(-1)
        pi = S.get('sinks', {}).get('pi', [])
        sink_node, sink_k = S.get('sinks', {}).get('node', []), S.get('sinks', {}).get('k', [])
        for h in range(H):
            disc = 0.99 ** h
            for d in range(nd):
                forecast = self.val(fc[d], h) if d < fc.shape[0] else 0.0
                demand = max(0.0, forecast) + (max(0.0, self.val(backlog, d)) if h == 0 else 0.0)
                bounds.append((0.0, demand))
                obj[yid(h, d)] = -max(0.0, self.val(pi, d, 1.0)) * disc
            for s in range(ns):
                obj[xid(h, s)] = costs[s] * disc

        available = np.zeros((nn, nk))
        stock = self.arr(obs, 'stock.qty', []).reshape(-1)
        for i, pair in enumerate(L.get('stock_slots', [])):
            if i < len(stock) and len(pair) >= 2:
                n, k = int(pair[0]), int(pair[1])
                if 0 <= n < nn and 0 <= k < nk:
                    available[n, k] += max(0.0, self.val(stock, i))
        supply = self.arr(obs, 'graph_now.supply.avail', []).reshape(-1)
        for i, pair in enumerate(L.get('supply_slots', [])):
            if i < len(supply) and len(pair) >= 2:
                n, k = int(pair[0]), int(pair[1])
                if 0 <= n < nn and 0 <= k < nk:
                    available[n, k] += max(0.0, self.val(supply, i))

        arrivals = np.zeros((H, nn, nk))
        pe = self.arr(obs, 'pipeline.edge', []).reshape(-1)
        pk = self.arr(obs, 'pipeline.k', []).reshape(-1)
        pq = self.arr(obs, 'pipeline.qty', []).reshape(-1)
        pa = self.arr(obs, 'pipeline.arrival_week', []).reshape(-1)
        for i in range(min(len(pe), len(pk), len(pq), len(pa))):
            e, k = int(pe[i]), int(pk[i])
            h = int(pa[i]) - week
            if 0 <= h < H and 0 <= e < ne and 0 <= k < nk:
                arrivals[h, int(E['head'][e]), k] += max(0.0, self.val(pq, i))

        A, b = [], []
        # Cumulative conservation permits inventory carryover within the horizon.
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
                            if q + delays[s] <= h and heads[s] == n:
                                row[xid(q, s)] -= 1.0
                        for d in range(nd):
                            if d < len(sink_node) and d < len(sink_k) and int(sink_node[d]) == n and int(sink_k[d]) == k:
                                row[yid(q, d)] += 1.0
                    if np.any(row):
                        A.append(row)
                        b.append(available[n, k] + float(np.sum(arrivals[:h + 1, n, k])))

        cp_nodes = list(L.get('chokepoints', []))
        cp_open = self.arr(obs, 'graph_now.open', np.ones(len(cp_nodes))).reshape(-1)
        ktb = self.arr(obs, 'graph_now.kappa.tb', np.full(len(cp_nodes), 1e12)).reshape(-1)
        kct = self.arr(obs, 'graph_now.kappa.ct', np.full(len(cp_nodes), 1e12)).reshape(-1)
        pools = S.get('commodities', {}).get('pool', ['tb'] * nk)
        for h in range(H):
            for e in range(ne):
                row = np.zeros(nv)
                for s in range(ns):
                    if e in paths[s]:
                        row[xid(h, s)] += 1.0
                if np.any(row):
                    A.append(row)
                    b.append(max(0.0, self.val(u, e, self.val(E.get('u0', []), e, 1e12))))
            for j, node in enumerate(cp_nodes):
                for pool, kap in (('tb', ktb), ('ct', kct)):
                    row = np.zeros(nv)
                    for s in range(ns):
                        k = int(slot_k[s])
                        if k < len(pools) and pools[k] == pool and any(int(E['head'][x]) == int(node) for x in paths[s]):
                            row[xid(h, s)] += 1.0
                    if np.any(row):
                        A.append(row)
                        b.append(max(0.0, self.val(cp_open, j, 1.0)) * max(0.0, self.val(kap, j, 1e12)))

        try:
            res = linprog(obj, A_ub=np.asarray(A), b_ub=np.asarray(b), bounds=bounds, method='highs')
            flows = np.maximum(0.0, res.x[:ns]) if res.success else zero
        except Exception:
            flows = zero
        return {'flows': flows.astype(float)}