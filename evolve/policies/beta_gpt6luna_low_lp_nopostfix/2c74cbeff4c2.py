# -2.1111972654833036
import numpy as np
from scipy.optimize import linprog


class Agent:
    def __init__(self, config):
        self.config = config
        self.static = config['static']
        self.layout = config['layout']
        self.slots = self.static['action_slots']
        self.edges = self.static['edges']
        self.lanes = self.static['lanes']
        self.n_edges = len(self.edges['tail'])
        self.n_slots = len(self.slots['edge'])
        self.n_demands = len(self.layout['demands'])

    @staticmethod
    def _num(x, i, default=0.0):
        try:
            v = x[i]
            if v is None:
                return default
            v = float(v)
            return v if np.isfinite(v) else default
        except (IndexError, TypeError, ValueError):
            return default

    def act(self, obs):
        S, L = self.static, self.layout
        edges, slots = S['edges'], S['action_slots']
        nodes_n = len(S['nodes']['id'])
        commodities_n = len(S['commodities']['id'])
        slot_n = len(slots['edge'])
        dem_n = len(L['demands'])
        if slot_n == 0 or dem_n == 0:
            return {'flows': np.zeros(slot_n, dtype=float)}

        # Build action routes and their edge incidence, costs and effective bounds.
        routes = []
        route_cost = []
        route_ub = []
        route_edge_use = []
        route_cp_use = []
        cp_nodes = list(L.get('chokepoints', []))
        cp_row = {int(n): i for i, n in enumerate(cp_nodes)}
        cp_caps = np.zeros(len(cp_nodes), dtype=float)
        open_v = np.asarray(obs.get('graph_now.open', np.ones(len(cp_nodes))), dtype=float).reshape(-1)
        k_tb = np.asarray(obs.get('graph_now.kappa.tb', np.full(len(cp_nodes), np.inf)), dtype=float).reshape(-1)
        k_ct = np.asarray(obs.get('graph_now.kappa.ct', np.full(len(cp_nodes), np.inf)), dtype=float).reshape(-1)
        cp_pool = S['commodities'].get('pool', ['tb'] * commodities_n)
        for j, node in enumerate(cp_nodes):
            kap = k_tb if j < len(k_tb) else k_ct
            cp_caps[j] = max(0.0, self._num(open_v, j, 1.0)) * max(0.0, self._num(kap, j, 1e12))

        u = np.asarray(obs.get('graph_now.u', edges.get('u0', [])), dtype=float).reshape(-1)
        c = np.asarray(obs.get('graph_now.c', edges.get('c0', [])), dtype=float).reshape(-1)
        tau = np.asarray(obs.get('graph_now.tau', edges.get('tau0', [])), dtype=float).reshape(-1)
        tariff = np.asarray(obs.get('graph_now.tariff', np.zeros((self.n_edges, commodities_n))), dtype=float)
        prohibited = np.asarray(obs.get('graph_now.prohibited', np.zeros((self.n_edges, commodities_n))), dtype=float)
        mask = np.asarray(obs.get('action_mask', np.ones(slot_n)), dtype=float).reshape(-1)
        slot_mask = np.asarray(obs.get('slot_mask', np.zeros(slot_n)), dtype=float).reshape(-1)
        slot_tail, slot_head, slot_commodity = [], [], []

        for s in range(slot_n):
            edge = int(slots['edge'][s])
            k = int(slots['k'][s])
            lane_val = slots.get('lane', [None] * slot_n)[s]
            lane = None if lane_val is None else int(lane_val)
            path = [edge]
            if lane is not None and 0 <= lane < len(self.lanes['edges']):
                path = [int(e) for e in self.lanes['edges'][lane]]
            if not path or edge < 0 or edge >= self.n_edges or k < 0 or k >= commodities_n:
                routes.append([]); route_cost.append(0.0); route_ub.append(0.0); route_edge_use.append([]); route_cp_use.append([])
                slot_tail.append(0); slot_head.append(0); slot_commodity.append(max(0, min(k, commodities_n-1)))
                continue
            tail = int(edges['tail'][path[0]])
            head = int(edges['head'][path[-1]])
            slot_tail.append(tail); slot_head.append(head); slot_commodity.append(k)
            ub = float('inf')
            cost = 0.0
            valid = True
            edge_use = []
            cp_use = []
            for e in path:
                if e < 0 or e >= self.n_edges:
                    valid = False
                    break
                if k < len(edges.get('K', [])) and edges['K'][e] and k not in edges['K'][e]:
                    valid = False
                if k < prohibited.shape[1] and e < prohibited.shape[0] and prohibited[e, k] > 0:
                    valid = False
                if e < len(u) and np.isfinite(u[e]):
                    ub = min(ub, max(0.0, float(u[e])))
                edge_use.append(e)
                cost += max(0.0, self._num(c, e, self._num(edges.get('c0', []), e)))
                if e < tariff.shape[0] and k < tariff.shape[1]:
                    cost += max(0.0, float(tariff[e, k])) * max(0.0, self._num(S['commodities'].get('v', []), k))
                # A route passing a chokepoint consumes its pool's weekly throughput.
                h = int(edges['head'][e])
                if h in cp_row:
                    j = cp_row[h]
                    pool = cp_pool[k] if k < len(cp_pool) else 'tb'
                    capvec = k_ct if pool == 'ct' else k_tb
                    cap = max(0.0, self._num(open_v, j, 1.0)) * max(0.0, self._num(capvec, j, 1e12))
                    cp_use.append((j, cap))
            if len(mask) > s and mask[s] <= 0 and len(slot_mask) > s and slot_mask[s] > 0:
                valid = False
            if not valid:
                ub = 0.0
            if not np.isfinite(ub):
                ub = 1e12
            routes.append(edge_use); route_cost.append(cost); route_ub.append(max(0.0, ub)); route_edge_use.append(edge_use); route_cp_use.append(cp_use)

        # Current usable inventory/supply, indexed by node and commodity.
        available = np.zeros((nodes_n, commodities_n), dtype=float)
        stock = np.asarray(obs.get('stock.qty', []), dtype=float).reshape(-1)
        for i, pair in enumerate(L.get('stock_slots', [])):
            if len(pair) >= 2 and i < len(stock):
                n, k = int(pair[0]), int(pair[1])
                if 0 <= n < nodes_n and 0 <= k < commodities_n:
                    available[n, k] += max(0.0, stock[i])
        supply = np.asarray(obs.get('graph_now.supply.avail', []), dtype=float).reshape(-1)
        for i, pair in enumerate(L.get('supply_slots', [])):
            if len(pair) >= 2 and i < len(supply):
                n, k = int(pair[0]), int(pair[1])
                if 0 <= n < nodes_n and 0 <= k < commodities_n:
                    available[n, k] += max(0.0, supply[i])

        # Shipments arriving this week can be used to meet this week's needs.
        pe = np.asarray(obs.get('pipeline.edge', []), dtype=int).reshape(-1)
        pk = np.asarray(obs.get('pipeline.k', []), dtype=int).reshape(-1)
        pq = np.asarray(obs.get('pipeline.qty', []), dtype=float).reshape(-1)
        pa = np.asarray(obs.get('pipeline.arrival_week', []), dtype=int).reshape(-1)
        week = int(np.asarray(obs.get('week', [1])).reshape(-1)[0])
        for i in range(min(len(pe), len(pk), len(pq), len(pa))):
            e, k = int(pe[i]), int(pk[i])
            if pa[i] == week and 0 <= e < self.n_edges and 0 <= k < commodities_n:
                n = int(edges['head'][e])
                if 0 <= n < nodes_n:
                    available[n, k] += max(0.0, float(pq[i]))

        # LP variables: route dispatches followed by demand served quantities.
        nv = slot_n + dem_n
        objective = np.zeros(nv, dtype=float)
        objective[:slot_n] = route_cost
        sinks_node = S.get('sinks', {}).get('node', [])
        sinks_k = S.get('sinks', {}).get('k', [])
        penalties = S.get('sinks', {}).get('pi', [])
        backlog = np.asarray(obs.get('backlog.qty', np.zeros(dem_n)), dtype=float).reshape(-1)
        forecast = np.asarray(obs.get('demand_forecast.qty', np.zeros((dem_n, 1))), dtype=float)
        dmax = np.zeros(dem_n, dtype=float)
        for d in range(dem_n):
            demand = max(0.0, self._num(backlog, d))
            if forecast.ndim == 2 and d < forecast.shape[0] and forecast.shape[1] > 0:
                demand += max(0.0, float(forecast[d, 0]))
            dmax[d] = demand
            objective[slot_n + d] = -max(0.0, self._num(penalties, d, 1.0))

        A, b = [], []
        # Node/commodity available-resource constraints: dispatch less incoming plus service
        # cannot exceed starting resources. This permits multi-hop LP routing.
        for n in range(nodes_n):
            for k in range(commodities_n):
                row = np.zeros(nv)
                for s in range(slot_n):
                    if slot_commodity[s] != k:
                        continue
                    if slot_tail[s] == n: row[s] += 1.0
                    if slot_head[s] == n: row[s] -= 1.0
                for d in range(dem_n):
                    if d < len(sinks_node) and d < len(sinks_k) and int(sinks_node[d]) == n and int(sinks_k[d]) == k:
                        row[slot_n + d] += 1.0
                if np.any(row):
                    A.append(row); b.append(available[n, k])
        for d in range(dem_n):
            row = np.zeros(nv); row[slot_n + d] = 1.0
            A.append(row); b.append(dmax[d])

        # Shared edge capacity across commodities and duplicate action routes.
        for e in range(self.n_edges):
            cap = self._num(u, e, 1e12)
            if not np.isfinite(cap): cap = 1e12
            row = np.zeros(nv)
            for s in range(slot_n):
                if e in route_edge_use[s]: row[s] += 1.0
            if np.any(row): A.append(row); b.append(max(0.0, cap))
        # Throughput limits for each chokepoint and pool, to avoid overbooking a route.
        for j, node in enumerate(cp_nodes):
            for pool in ('tb', 'ct'):
                capvec = k_tb if pool == 'tb' else k_ct
                cap = max(0.0, self._num(open_v, j, 1.0)) * max(0.0, self._num(capvec, j, 1e12))
                row = np.zeros(nv)
                for s in range(slot_n):
                    k = slot_commodity[s]
                    if cp_pool[k] != pool: continue
                    if any(int(edges['head'][e]) == int(node) for e in route_edge_use[s]): row[s] += 1.0
                if np.any(row): A.append(row); b.append(cap)

        bounds = [(0.0, route_ub[s]) for s in range(slot_n)] + [(0.0, dmax[d]) for d in range(dem_n)]
        try:
            result = linprog(objective, A_ub=np.asarray(A), b_ub=np.asarray(b), bounds=bounds, method='highs')
            flows = np.maximum(0.0, result.x[:slot_n]) if result.success else np.zeros(slot_n)
        except Exception:
            flows = np.zeros(slot_n)
        return {'flows': flows.astype(float)}