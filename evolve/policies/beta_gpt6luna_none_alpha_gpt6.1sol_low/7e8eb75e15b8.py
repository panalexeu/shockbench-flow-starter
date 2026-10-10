# -2.1085142492894287
import numpy as np

try:
    from scipy.optimize import linprog
except Exception:
    linprog = None


class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.T = int(config['T'])
        self.slots = self.s['action_slots']
        self.edges = self.s['edges']
        self.tail = np.asarray(self.edges['tail'], dtype=int)
        self.head = np.asarray(self.edges['head'], dtype=int)
        self.nedge = len(self.tail)
        self.nslot = len(self.slots['edge'])
        self.v = np.asarray(self.s['commodities']['v'], dtype=float)
        self.pool = self.s['commodities'].get('pool', [])
        self.cids = self.s['commodities']['id']
        self.nk = len(self.cids)
        self.cp_row = {int(n): i for i, n in enumerate(self.l.get('chokepoints', []))}
        self.stock_row = {tuple(map(int, x)): i for i, x in enumerate(self.l.get('stock_slots', []))}
        self.supply_row = {tuple(map(int, x)): i for i, x in enumerate(self.l.get('supply_slots', []))}
        self.demand_row = {tuple(map(int, x)): i for i, x in enumerate(self.l.get('demands', []))}
        self.sink_nodes = np.asarray(self.s.get('sinks', {}).get('node', []), dtype=int)
        self.sink_ks = np.asarray(self.s.get('sinks', {}).get('k', []), dtype=int)
        self.sink_pi = np.asarray(self.s.get('sinks', {}).get('pi', []), dtype=float)
        self.sink_backlog = np.asarray(self.s.get('sinks', {}).get('backlog', []), dtype=bool)
        self.lane_edges = self.s.get('lanes', {}).get('edges', [])
        self.routes = []
        for i, (e0, k0, lane0) in enumerate(zip(self.slots['edge'], self.slots['k'], self.slots['lane'])):
            e, k = int(e0), int(k0)
            es = [e] if lane0 is None or int(lane0) < 0 else list(map(int, self.lane_edges[int(lane0)]))
            if not es:
                es = [e]
            src, dst = int(self.tail[es[0]]), int(self.head[es[-1]])
            self.routes.append((i, src, dst, k, es))
        self.lot_keys = self.l.get('lot_keys', [])
        self.slot_edge = np.asarray(self.slots['edge'], dtype=int)

    def _dest(self, edge, lane):
        if lane >= 0 and lane < len(self.lane_edges):
            es = self.lane_edges[lane]
            if es:
                return int(self.head[int(es[-1])])
        return int(self.head[edge])

    def _fallback(self, o):
        flows = np.zeros(self.nslot, dtype=float)
        u = np.asarray(o['graph_now.u'], dtype=float)
        mask = np.asarray(o['action_mask'], dtype=bool)
        supply = {}
        for key, row in self.supply_row.items():
            supply[key] = max(0.0, float(o['graph_now.supply.avail'][row]))
        for key, row in self.stock_row.items():
            supply[key] = supply.get(key, 0.0) + max(0.0, float(o['stock.qty'][row]))
        demand = {}
        for key, row in self.demand_row.items():
            f = np.asarray(o['demand_forecast.qty'][row], dtype=float)
            fm = o.get('demand_forecast.qty.observed')
            if fm is not None:
                f = f[np.asarray(fm[row], dtype=bool)]
            rate = max(0.0, float(np.mean(f))) if len(f) else 0.0
            demand[key] = rate + max(0.0, float(o['backlog.qty'][row]))
        edge_left = np.maximum(0.0, u.copy())
        candidates = []
        for i, src, dst, k, es in self.routes:
            if not mask[i] or (dst, k) not in demand:
                continue
            cap = min([max(0.0, float(u[e])) for e in es] or [0.0])
            if cap <= 0:
                continue
            cost = sum(float(o['graph_now.c'][e]) + float(o['graph_now.tariff'][e, k]) * self.v[k] for e in es)
            candidates.append((cost, i, src, dst, k, es, cap))
        for cost, i, src, dst, k, es, cap in sorted(candidates):
            need = demand[(dst, k)]
            have = supply.get((dst, k), 0.0)
            qty = min(cap, max(0.0, need - have), supply.get((src, k), 0.0))
            for e in es:
                qty = min(qty, edge_left[e])
            if qty <= 0:
                continue
            flows[i] = qty
            supply[(src, k)] -= qty
            supply[(dst, k)] = supply.get((dst, k), 0.0) + qty
            demand[(dst, k)] -= qty
            for e in es:
                edge_left[e] -= qty
        return {'flows': flows}

    def act(self, observation):
        o = observation
        if linprog is None:
            return self._fallback(o)
        week = int(o['week'][0])
        nnode = len(self.s['nodes']['id'])
        nk = self.nk
        nstate = nnode * nk
        # Optimize the present dispatch and a short continuation plan. Forecasts and
        # current disruptions are known; future capacity is conservatively nominal.
        H = min(max(2, max((int(x) for x in self.edges['tau0'] if x is not None), default=1) + 2), max(1, self.T - week + 1), 6)
        u_now = np.asarray(o['graph_now.u'], dtype=float)
        c_now = np.asarray(o['graph_now.c'], dtype=float)
        tau_now = np.asarray(o['graph_now.tau'], dtype=float)
        tariff_now = np.asarray(o['graph_now.tariff'], dtype=float)
        prohibited = np.asarray(o['graph_now.prohibited'], dtype=bool)
        slotmask = np.asarray(o['action_mask'], dtype=bool)
        open_now = np.asarray(o['graph_now.open'], dtype=float)
        # Route variables are created for each dispatch week and action slot.
        rvars = []
        for h in range(H):
            for i, src, dst, k, es in self.routes:
                if h == 0 and (not slotmask[i]):
                    continue
                if any(prohibited[e, k] for e in es) and h == 0:
                    continue
                tau = sum(max(0, int(tau_now[e])) if h == 0 else max(0, int(self.edges['tau0'][e] or 0)) for e in es)
                arrival = h + max(1, tau)
                if arrival >= H:
                    continue
                cap = float('inf')
                for e in es:
                    if self.edges['u0'][e] is not None:
                        cap = min(cap, max(0.0, float(u_now[e]) if h == 0 else float(self.edges['u0'][e])))
                if not np.isfinite(cap):
                    cap = 1e6
                if h == 0:
                    for e in es:
                        cp = self.cp_row.get(int(self.head[e]))
                        if cp is not None:
                            cap *= max(0.0, float(open_now[cp]))
                if cap <= 1e-9:
                    continue
                cost = 0.0
                for e in es:
                    freight = float(c_now[e]) if h == 0 else float(self.edges['c0'][e] or 0.0)
                    tr = float(tariff_now[e, k]) if h == 0 else 0.0
                    cost += freight + tr * self.v[k]
                cost += 0.01 * self.v[k] * max(0, tau)
                rvars.append((h, i, src, dst, k, es, arrival, cap, cost))
        if not rvars:
            return {'flows': np.zeros(self.nslot, dtype=float)}
        # Variables: route dispatches, sink unmet amounts in each modeled week.
        sink_vars = []
        for h in range(H):
            for d, (node, k) in enumerate(zip(self.sink_nodes, self.sink_ks)):
                sink_vars.append((h, d, int(node), int(k)))
        nr, ns = len(rvars), len(sink_vars)
        nv = nr + ns
        cobj = np.zeros(nv, dtype=float)
        for j, rv in enumerate(rvars):
            cobj[j] = rv[8] + 1e-5
        demand_amount = np.zeros(ns, dtype=float)
        for z, (h, d, node, k) in enumerate(sink_vars):
            row = self.demand_row.get((node, k))
            if row is None:
                continue
            forecast = np.asarray(o['demand_forecast.qty'][row], dtype=float)
            fm = o.get('demand_forecast.qty.observed')
            if fm is not None:
                valid = np.asarray(fm[row], dtype=bool)
                vals = forecast[valid]
            else:
                vals = forecast
            dem = max(0.0, float(forecast[min(h, len(forecast)-1)])) if len(forecast) else 0.0
            if h == 0:
                dem += max(0.0, float(o['backlog.qty'][row]))
            demand_amount[z] = dem
            pi = float(self.sink_pi[d]) if d < len(self.sink_pi) else float(np.max(self.v))
            cobj[nr + z] = -max(1.0, pi)
        # State balance: for each node/commodity/time, starting resources plus arrivals
        # must cover departures and sink consumption. Supply enters each week.
        Aeq, beq = [], []
        init = np.zeros((H, nstate), dtype=float)
        for key, row in self.stock_row.items():
            node, k = key
            init[0, node * nk + k] += max(0.0, float(o['stock.qty'][row]))
        for key, row in self.supply_row.items():
            node, k = key
            init[0, node * nk + k] += max(0.0, float(o['graph_now.supply.avail'][row]))
        # Include in-transit shipments arriving in modeled weeks.
        pq = o.get('pipeline.qty', np.zeros(0))
        pm = o.get('pipeline.qty.observed', np.ones_like(pq, dtype=np.int8))
        plm = o.get('pipeline.lane.observed', np.ones_like(pm, dtype=np.int8))
        for j in np.flatnonzero(pm):
            q = max(0.0, float(pq[j]))
            if q <= 0:
                continue
            arr = int(o['pipeline.arrival_week'][j]) - week
            if arr < 0 or arr >= H:
                continue
            e, k = int(o['pipeline.edge'][j]), int(o['pipeline.k'][j])
            lane = int(o['pipeline.lane'][j]) if plm[j] else -1
            node = self._dest(e, lane)
            init[arr, node * nk + k] += q
        # Queued chokepoint cargo arrives when released; count as soon-arriving stock.
        for r, key in enumerate(self.lot_keys):
            cp, k, lane, edge = map(int, key)
            qrow = np.asarray(o.get('queue_lots.qty', np.zeros((0, 0)))[r], dtype=float) if r < len(o.get('queue_lots.qty', [])) else np.zeros(0)
            q = float(np.sum(qrow))
            if q > 0:
                node = self._dest(edge, lane)
                init[0, node * nk + k] += q
        # Balance equations use time-indexed inventory variables, enabling holding.
        inv0 = nr + ns
        inv_index = np.arange(inv0, inv0 + H * nstate).reshape(H, nstate)
        nv2 = inv0 + H * nstate
        cfull = np.zeros(nv2, dtype=float)
        cfull[:nv] = cobj
        # Small holding cost avoids pointless accumulation while allowing buffers.
        cfull[inv0:] = 1e-4
        eqrows = []
        beqs = []
        for h in range(H):
            for st in range(nstate):
                row = np.zeros(nv2, dtype=float)
                row[inv_index[h, st]] = 1.0
                if h > 0:
                    row[inv_index[h-1, st]] = -1.0
                b = init[h, st]
                for j, rv in enumerate(rvars):
                    rh, i, src, dst, k, es, arrival, cap, cost = rv
                    if k * nnode + src == st and rh == h:
                        row[j] += 1.0
                    if k * nnode + dst == st and arrival == h:
                        row[j] -= 1.0
                for z, (sh, d, node, k) in enumerate(sink_vars):
                    if sh == h and node * nk + k == st:
                        # demand served equals demand less unmet; negative unmet coefficient
                        row[nr + z] += 1.0
                        b -= demand_amount[z]
                eqrows.append(row)
                beqs.append(b)
        # Capacity rows: edge and chokepoint pool per dispatch week, plus route bounds.
        aub, bub = [], []
        for h in range(H):
            for e in range(self.nedge):
                cap = max(0.0, float(u_now[e]) if h == 0 else float(self.edges['u0'][e] or 0.0))
                ids = [j for j, rv in enumerate(rvars) if rv[0] == h and e in rv[5]]
                if ids:
                    row = np.zeros(nv2, dtype=float)
                    row[ids] = 1.0
                    aub.append(row); bub.append(cap)
            for cpnode, cp in self.cp_row.items():
                for pool in ('tb', 'ct'):
                    arrname = 'graph_now.kappa.tb' if pool == 'tb' else 'graph_now.kappa.ct'
                    cap = max(0.0, float(o[arrname][cp])) if h == 0 else 1e6
                    ids = []
                    for j, rv in enumerate(rvars):
                        if rv[0] != h or self.pool[rv[4]] != pool:
                            continue
                        if any(int(self.head[e]) == cpnode for e in rv[5]):
                            ids.append(j)
                    if ids:
                        row = np.zeros(nv2, dtype=float); row[ids] = 1.0
                        aub.append(row); bub.append(cap)
        bounds = [(0, rv[7]) for rv in rvars] + [(0, demand_amount[z]) for z in range(ns)] + [(0, None)] * (H * nstate)
        try:
            res = linprog(cfull, A_ub=np.asarray(aub) if aub else None, b_ub=np.asarray(bub) if bub else None,
                          A_eq=np.asarray(eqrows), b_eq=np.asarray(beqs), bounds=bounds, method='highs')
        except Exception:
            return self._fallback(o)
        if not res.success or res.x is None:
            return self._fallback(o)
        flows = np.zeros(self.nslot, dtype=float)
        for j, rv in enumerate(rvars):
            if rv[0] == 0:
                flows[rv[1]] += max(0.0, float(res.x[j]))
        return {'flows': flows}
