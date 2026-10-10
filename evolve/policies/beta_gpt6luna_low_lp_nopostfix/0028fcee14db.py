# -2.107347404137677
import numpy as np
from scipy.optimize import linprog


class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.edges = self.s['edges']
        self.slots = self.s['action_slots']
        self.lanes = self.s.get('lanes', {})
        self.ne = len(self.edges.get('tail', []))
        self.nn = len(self.s.get('nodes', {}).get('id', []))
        self.nk = len(self.s.get('commodities', {}).get('id', []))
        self.ns = len(self.slots.get('edge', []))
        self.nd = len(self.l.get('demands', []))

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
        if ns == 0 or nd == 0 or nn == 0 or nk == 0:
            return {'flows': np.zeros(ns, dtype=float)}
        S, L, E = self.s, self.l, self.edges
        week = int(np.asarray(obs.get('week', [1])).reshape(-1)[0])
        T = int(self.s.get('T', self.s.get('instance', {}).get('T', 52)))
        fc = np.asarray(obs.get('demand_forecast.qty', np.zeros((nd, 1))), dtype=float)
        if fc.ndim != 2:
            fc = np.zeros((nd, 1))
        H = max(1, min(8, fc.shape[1], T - week + 1))

        u = np.asarray(obs.get('graph_now.u', E.get('u0', [])), dtype=float).reshape(-1)
        c = np.asarray(obs.get('graph_now.c', E.get('c0', [])), dtype=float).reshape(-1)
        tau = np.asarray(obs.get('graph_now.tau', E.get('tau0', [])), dtype=float).reshape(-1)
        tariff = np.asarray(obs.get('graph_now.tariff', np.zeros((ne, nk))), dtype=float)
        prohibited = np.asarray(obs.get('graph_now.prohibited', np.zeros((ne, nk))), dtype=float)
        mask = np.asarray(obs.get('action_mask', np.ones(ns)), dtype=float).reshape(-1)
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
            cost, delay, cap = 0.0, 0, 1e12
            if valid:
                for x in path:
                    cap = min(cap, max(0.0, self.num(u, x, self.num(E.get('u0', []), x, 1e12))))
                    cost += max(0.0, self.num(c, x, self.num(E.get('c0', []), x)))
                    if x < tariff.shape[0] and k < tariff.shape[1]:
                        cost += max(0.0, self.num(tariff[x], k)) * max(0.0, self.num(values, k))
                    delay += max(0, int(self.num(tau, x, self.num(E.get('tau0', []), x))))
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

        nx, ny = H * ns, H * nd
        nv = nx + ny
        xid = lambda h, s: h * ns + s
        yid = lambda h, d: nx + h * nd + d
        obj = np.zeros(nv)
        bounds = [(0.0, caps[s]) for h in range(H) for s in range(ns)]
        demand = np.zeros((H, nd))
        backlog = np.asarray(obs.get('backlog.qty', np.zeros(nd)), dtype=float).reshape(-1)
        pi = S.get('sinks', {}).get('pi', [])
        for h in range(H):
            for d in range(nd):
                f = max(0.0, self.num(fc[d], h)) if d < fc.shape[0] else 0.0
                demand[h, d] = f + (max(0.0, self.num(backlog, d)) if h == 0 else 0.0)
                bounds.append((0.0, demand[h, d]))
                discount = 0.99 ** h
                obj[yid(h, d)] = -max(0.0, self.num(pi, d, 1.0)) * discount
            for s in range(ns):
                obj[xid(h, s)] = costs[s] * (0.99 ** h)

        available = np.zeros((nn, nk))
        stock = np.asarray(obs.get('stock.qty', []), dtype=float).reshape(-1)
        for i, pair in enumerate(L.get('stock_slots', [])):
            if i < len(stock) and len(pair) >= 2:
                n, k = int(pair[0]), int(pair[1])
                if 0 <= n < nn and 0 <= k < nk:
                    available[n, k] += max(0.0, stock[i])
        supply = np.asarray(obs.get('graph_now.supply.avail', []), dtype=float).reshape(-1)
        for i, pair in enumerate(L.get('supply_slots', [])):
            if i < len(supply) and len(pair) >= 2:
                n, k = int(pair[0]), int(pair[1])
                if 0 <= n < nn and 0 <= k < nk:
                    available[n, k] += max(0.0, supply[i])

        arrivals = np.zeros((H, nn, nk))
        pe = np.asarray(obs.get('pipeline.edge', []), dtype=int).reshape(-1)
        pk = np.asarray(obs.get('pipeline.k', []), dtype=int).reshape(-1)
        pq = np.asarray(obs.get('pipeline.qty', []), dtype=float).reshape(-1)
        pa = np.asarray(obs.get('pipeline.arrival_week', []), dtype=int).reshape(-1)
        for i in range(min(len(pe), len(pk), len(pq), len(pa))):
            e, k = int(pe[i]), int(pk[i])
            h = int(pa[i]) - week
            if 0 <= h < H and 0 <= e < ne and 0 <= k < nk:
                arrivals[h, int(E['head'][e]), k] += max(0.0, self.num(pq, i))

        A, b = [], []
        sink_node = S.get('sinks', {}).get('node', [])
        sink_k = S.get('sinks', {}).get('k', [])
        # Cumulative inventory balances allow carryover, while shipments become
        # usable only after their modeled transit time.
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

        # Each edge's capacity is shared by all routes and commodities using it.
        for h in range(H):
            for e in range(ne):
                row = np.zeros(nv)
                for s in range(ns):
                    if e in paths[s]:
                        row[xid(h, s)] += 1.0
                if np.any(row):
                    A.append(row)
                    b.append(max(0.0, self.num(u, e, self.num(E.get('u0', []), e, 1e12))))

        try:
            res = linprog(obj, A_ub=np.asarray(A), b_ub=np.asarray(b), bounds=bounds, method='highs')
            flows = np.maximum(0.0, res.x[:ns]) if res.success else np.zeros(ns)
        except Exception:
            flows = np.zeros(ns)
        return {'flows': flows.astype(float)}