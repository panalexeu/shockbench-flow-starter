# -2.1079320372648676
import numpy as np
from scipy.optimize import linprog


class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.edges = self.s['edges']
        self.slots = self.s['action_slots']
        self.lanes = self.s.get('lanes', {})
        self.n_edges = len(self.edges.get('tail', []))
        self.n_nodes = len(self.s.get('nodes', {}).get('id', []))
        self.n_goods = len(self.s.get('commodities', {}).get('id', []))
        self.n_slots = len(self.slots.get('edge', []))

    @staticmethod
    def val(a, i, default=0.0):
        try:
            x = a[i]
            if x is None:
                return default
            x = float(x)
            return x if np.isfinite(x) else default
        except (IndexError, TypeError, ValueError):
            return default

    def act(self, obs):
        ns, nd, ng = self.n_slots, len(self.l.get('demands', [])), self.n_goods
        if ns == 0 or nd == 0 or ng == 0:
            return {'flows': np.zeros(ns, dtype=float)}
        S, E, L = self.s, self.edges, self.l
        week = int(np.asarray(obs.get('week', [1])).reshape(-1)[0])
        T = int(self.s.get('T', self.s.get('instance', {}).get('T', 52)))
        forecasts = np.asarray(obs.get('demand_forecast.qty', np.zeros((nd, 1))), dtype=float)
        # Forecast several weeks ahead, capped by the episode horizon and available forecast.
        H = max(1, min(8, forecasts.shape[1] if forecasts.ndim == 2 else 1, T - week + 1))
        if H < 1:
            return {'flows': np.zeros(ns, dtype=float)}

        slot_e = [int(x) for x in self.slots['edge']]
        slot_k = [int(x) for x in self.slots['k']]
        slot_lane = self.slots.get('lane', [None] * ns)
        paths, tails, heads, travel, costs, upper, allowed = [], [], [], [], [], [], []
        u = np.asarray(obs.get('graph_now.u', E.get('u0', [])), dtype=float).reshape(-1)
        c = np.asarray(obs.get('graph_now.c', E.get('c0', [])), dtype=float).reshape(-1)
        tau = np.asarray(obs.get('graph_now.tau', E.get('tau0', [])), dtype=float).reshape(-1)
        tariff = np.asarray(obs.get('graph_now.tariff', np.zeros((self.n_edges, ng))), dtype=float)
        prohibited = np.asarray(obs.get('graph_now.prohibited', np.zeros((self.n_edges, ng))), dtype=float)
        mask = np.asarray(obs.get('action_mask', np.ones(ns)), dtype=float).reshape(-1)
        commodities = S.get('commodities', {})
        values = commodities.get('v', [])
        edge_K = E.get('K', [])
        for s in range(ns):
            e, k = slot_e[s], slot_k[s]
            lv = slot_lane[s] if s < len(slot_lane) else None
            path = [e]
            if lv is not None:
                try:
                    path = [int(x) for x in self.lanes['edges'][int(lv)]]
                except (KeyError, IndexError, TypeError, ValueError):
                    path = []
            valid = bool(path) and 0 <= k < ng and all(0 <= x < self.n_edges for x in path)
            if valid:
                for x in path:
                    if x < len(edge_K) and edge_K[x] and k not in edge_K[x]:
                        valid = False
                    if x < prohibited.shape[0] and k < prohibited.shape[1] and prohibited[x, k] > 0:
                        valid = False
            if s < len(mask) and mask[s] <= 0:
                valid = False
            cap = 1e12
            cost = 0.0
            delay = 0
            for x in path if valid else []:
                if x < len(u) and np.isfinite(u[x]):
                    cap = min(cap, max(0.0, float(u[x])))
                cost += max(0.0, self.val(c, x, self.val(E.get('c0', []), x)))
                if x < tariff.shape[0] and k < tariff.shape[1]:
                    cost += max(0.0, float(tariff[x, k])) * max(0.0, self.val(values, k))
                delay += max(0, int(self.val(tau, x, self.val(E.get('tau0', []), x))))
            paths.append(path if valid else [])
            tails.append(int(E['tail'][path[0]]) if valid else 0)
            heads.append(int(E['head'][path[-1]]) if valid else 0)
            travel.append(max(1, delay))
            costs.append(cost)
            upper.append(cap if valid else 0.0)
            allowed.append(valid)

        # LP variables: route dispatches x[week,slot], followed by service y[week,demand].
        nx, ny = H * ns, H * nd
        nv = nx + ny
        xid = lambda h, s: h * ns + s
        yid = lambda h, d: nx + h * nd + d
        obj = np.zeros(nv)
        bounds = [(0.0, upper[s]) for h in range(H) for s in range(ns)]
        demand = np.zeros((H, nd))
        backlog = np.asarray(obs.get('backlog.qty', np.zeros(nd)), dtype=float).reshape(-1)
        sink_node = S.get('sinks', {}).get('node', [])
        sink_k = S.get('sinks', {}).get('k', [])
        penalties = S.get('sinks', {}).get('pi', [])
        for h in range(H):
            for d in range(nd):
                f = max(0.0, self.val(forecasts[d] if forecasts.ndim == 2 and d < forecasts.shape[0] and h < forecasts.shape[1] else [], h))
                demand[h, d] = f + (max(0.0, self.val(backlog, d)) if h == 0 else 0.0)
                # A small discount discourages unnecessary early dispatch while preserving future service value.
                discount = 0.98 ** h
                bounds.append((0.0, demand[h, d]))
                obj[yid(h, d)] = -max(0.0, self.val(penalties, d, 1.0)) * discount
            for s in range(ns):
                obj[xid(h, s)] = costs[s] * (0.98 ** h)

        # Inventory conservation at each node, commodity and time period.
        available = np.zeros((self.n_nodes, ng))
        stock = np.asarray(obs.get('stock.qty', []), dtype=float).reshape(-1)
        for i, pair in enumerate(L.get('stock_slots', [])):
            if i < len(stock) and len(pair) >= 2:
                n, k = int(pair[0]), int(pair[1])
                if 0 <= n < self.n_nodes and 0 <= k < ng:
                    available[n, k] += max(0.0, stock[i])
        supply = np.asarray(obs.get('graph_now.supply.avail', []), dtype=float).reshape(-1)
        for i, pair in enumerate(L.get('supply_slots', [])):
            if i < len(supply) and len(pair) >= 2:
                n, k = int(pair[0]), int(pair[1])
                if 0 <= n < self.n_nodes and 0 <= k < ng:
                    available[n, k] += max(0.0, supply[i])
        arrivals = np.zeros((H, self.n_nodes, ng))
        pe = np.asarray(obs.get('pipeline.edge', []), dtype=int).reshape(-1)
        pk = np.asarray(obs.get('pipeline.k', []), dtype=int).reshape(-1)
        pq = np.asarray(obs.get('pipeline.qty', []), dtype=float).reshape(-1)
        pa = np.asarray(obs.get('pipeline.arrival_week', []), dtype=int).reshape(-1)
        for i in range(min(len(pe), len(pk), len(pq), len(pa))):
            e, k = int(pe[i]), int(pk[i])
            h = int(pa[i]) - week
            if 0 <= h < H and 0 <= e < self.n_edges and 0 <= k < ng:
                n = int(E['head'][e])
                if 0 <= n < self.n_nodes:
                    arrivals[h, n, k] += max(0.0, float(pq[i]))

        A, b = [], []
        for h in range(H):
            for n in range(self.n_nodes):
                for k in range(ng):
                    row = np.zeros(nv)
                    for s in range(ns):
                        if slot_k[s] != k:
                            continue
                        if tails[s] == n:
                            row[xid(h, s)] += 1.0
                        # Shipments dispatched in earlier weeks become available at their destination on arrival.
                        for q in range(h):
                            if q + travel[s] == h and heads[s] == n:
                                row[xid(q, s)] -= 1.0
                    for d in range(nd):
                        if d < len(sink_node) and d < len(sink_k) and int(sink_node[d]) == n and int(sink_k[d]) == k:
                            row[yid(h, d)] += 1.0
                    if np.any(row):
                        rhs = (available[n, k] if h == 0 else 0.0) + arrivals[h, n, k]
                        A.append(row); b.append(rhs)
        # Demand upper bounds are also represented by variable bounds; cap shared edge usage each week.
        for h in range(H):
            for e in range(self.n_edges):
                cap = max(0.0, self.val(u, e, self.val(E.get('u0', []), e, 1e12)))
                row = np.zeros(nv)
                for s in range(ns):
                    if e in paths[s]:
                        row[xid(h, s)] += 1.0
                if np.any(row):
                    A.append(row); b.append(cap)
        try:
            res = linprog(obj, A_ub=np.asarray(A), b_ub=np.asarray(b), bounds=bounds, method='highs')
            flows = np.maximum(0.0, res.x[:ns]) if res.success else np.zeros(ns)
        except Exception:
            flows = np.zeros(ns)
        return {'flows': flows.astype(float)}