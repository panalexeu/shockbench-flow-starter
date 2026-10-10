# -2.107836226986223
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
    def arr(obs, key, default):
        try:
            return np.asarray(obs.get(key, default))
        except Exception:
            return np.asarray(default)

    @staticmethod
    def num(a, i, default=0.0):
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
        zero = np.zeros(ns, dtype=float)
        if not ns or not nd or not nn or not nk:
            return {'flows': zero}
        S, L, E = self.S, self.L, self.E
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
        slot_e, slot_k = self.slots['edge'], self.slots['k']
        slot_lane = self.slots.get('lane', [None] * ns)
        edge_K = E.get('K', [])
        values = S.get('commodities', {}).get('v', [])

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
                    cap = min(cap, max(0.0, self.num(u, x, self.num(E.get('u0', []), x, 1e12))))
                    cost += max(0.0, self.num(c, x, self.num(E.get('c0', []), x)))
                    try:
                        if x < tariff.shape[0] and k < tariff.shape[1]:
                            cost += max(0.0, float(tariff[x, k])) * max(0.0, self.num(values, k))
                    except (IndexError, TypeError, ValueError):
                        pass
                    delay += max(0, int(self.num(tau, x, self.num(E.get('tau0', []), x))))
                tail, head = int(E['tail'][path[0]]), int(E['head'][path[-1]])
            else:
                path, cap, tail, head = [], 0.0, -1, -1
            paths.append(path)
            tails.append(tail)
            heads.append(head)
            delays.append(max(1, delay))
            costs.append(cost)
            caps.append(cap)

        nx, nv = H * ns, H * ns + H * nd
        xid = lambda h, s: h * ns + s
        yid = lambda h, d: nx + h * nd + d
        obj = np.zeros(nv)
        bounds = [(0.0, caps[s]) for h in range(H) for s in range(ns)]
        backlog = self.arr(obs, 'backlog.qty', np.zeros(nd)).reshape(-1)
        penalties = S.get('sinks', {}).get('pi', [])
        for h in range(H):
            # Future planning helps position inventory, but should not dominate the
            # action taken now when future conditions are not known.
            discount = 0.90 ** h
            for d in range(nd):
                f = max(0.0, self.num(fc[d], h)) if d < fc.shape[0] else 0.0
                dem = f + (max(0.0, self.num(backlog, d)) if h == 0 else 0.0)
                bounds.append((0.0, dem))
                obj[yid(h, d)] = -max(0.0, self.num(penalties, d, 1.0)) * discount
            for s in range(ns):
                # Tiny future dispatch penalty breaks ties in favor of acting now.
                obj[xid(h, s)] = costs[s] * discount + (1e-5 * (h + 1) if h else 0.0)

        available = np.zeros((nn, nk))
        stock = self.arr(obs, 'stock.qty', []).reshape(-1)
        for i, pair in enumerate(L.get('stock_slots', [])):
            if i < len(stock) and len(pair) >= 2:
                n, k = int(pair[0]), int(pair[1])
                if 0 <= n < nn and 0 <= k < nk:
                    available[n, k] += max(0.0, self.num(stock, i))
        supply = self.arr(obs, 'graph_now.supply.avail', []).reshape(-1)
        for i, pair in enumerate(L.get('supply_slots', [])):
            if i < len(supply) and len(pair) >= 2:
                n, k = int(pair[0]), int(pair[1])
                if 0 <= n < nn and 0 <= k < nk:
                    available[n, k] += max(0.0, self.num(supply, i))

        arrivals = np.zeros((H, nn, nk))
        pe = self.arr(obs, 'pipeline.edge', []).reshape(-1)
        pk = self.arr(obs, 'pipeline.k', []).reshape(-1)
        pq = self.arr(obs, 'pipeline.qty', []).reshape(-1)
        pa = self.arr(obs, 'pipeline.arrival_week', []).reshape(-1)
        for i in range(min(len(pe), len(pk), len(pq), len(pa))):
            e, k = int(pe[i]), int(pk[i])
            h = int(pa[i]) - week
            if 0 <= h < H and 0 <= e < ne and 0 <= k < nk:
                n = int(E['head'][e])
                if 0 <= n < nn:
                    arrivals[h, n, k] += max(0.0, self.num(pq, i))

        A, b = [], []
        sink_node = S.get('sinks', {}).get('node', [])
        sink_k = S.get('sinks', {}).get('k', [])
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
                    rhs = available[n, k] + float(np.sum(arrivals[:h + 1, n, k]))
                    if np.any(row):
                        A.append(row)
                        b.append(rhs)

        for h in range(H):
            for e in range(ne):
                row = np.zeros(nv)
                for s in range(ns):
                    if e in paths[s]:
                        row[xid(h, s)] += 1.0
                if np.any(row):
                    A.append(row)
                    b.append(max(0.0, self.num(u, e, self.num(E.get('u0', []), e, 1e12))))

        cps = list(L.get('chokepoints', []))
        opened = self.arr(obs, 'graph_now.open', np.ones(len(cps))).reshape(-1)
        ktb = self.arr(obs, 'graph_now.kappa.tb', np.full(len(cps), 1e12)).reshape(-1)
        kct = self.arr(obs, 'graph_now.kappa.ct', np.full(len(cps), 1e12)).reshape(-1)
        pools = S.get('commodities', {}).get('pool', ['tb'] * nk)
        for h in range(H):
            for j, node in enumerate(cps):
                for pool, kappas in (('tb', ktb), ('ct', kct)):
                    row = np.zeros(nv)
                    for s in range(ns):
                        k = int(slot_k[s])
                        if k < len(pools) and pools[k] == pool and any(int(E['head'][e]) == int(node) for e in paths[s]):
                            row[xid(h, s)] += 1.0
                    if np.any(row):
                        cap = max(0.0, self.num(opened, j, 1.0)) * max(0.0, self.num(kappas, j, 1e12))
                        A.append(row)
                        b.append(cap)

        try:
            res = linprog(obj, A_ub=np.asarray(A), b_ub=np.asarray(b), bounds=bounds, method='highs')
            flows = np.maximum(0.0, res.x[:ns]) if res.success else zero
        except Exception:
            flows = zero
        return {'flows': flows.astype(float)}