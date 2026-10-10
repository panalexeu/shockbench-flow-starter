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
        self.edges = self.s['edges']
        self.slots = self.s['action_slots']
        self.tail = np.asarray(self.edges['tail'], dtype=int)
        self.head = np.asarray(self.edges['head'], dtype=int)
        self.nnode = len(self.s['nodes']['id'])
        self.nk = len(self.s['commodities']['id'])
        self.nslot = len(self.slots['edge'])
        self.v = np.asarray(self.s['commodities']['v'], dtype=float)
        self.cp_rows = {int(n): i for i, n in enumerate(self.l.get('chokepoints', []))}
        self.stock_rows = {tuple(map(int, x)): i for i, x in enumerate(self.l.get('stock_slots', []))}
        self.supply_rows = {tuple(map(int, x)): i for i, x in enumerate(self.l.get('supply_slots', []))}
        self.demand_rows = {tuple(map(int, x)): i for i, x in enumerate(self.l.get('demands', []))}
        sinks = self.s.get('sinks', {})
        self.sink_nodes = np.asarray(sinks.get('node', []), dtype=int)
        self.sink_ks = np.asarray(sinks.get('k', []), dtype=int)
        self.sink_pi = np.asarray(sinks.get('pi', []), dtype=float)
        self.lane_edges = self.s.get('lanes', {}).get('edges', [])
        self.lot_keys = self.l.get('lot_keys', [])
        self.routes = []
        for i, (e0, k0, lane0) in enumerate(zip(self.slots['edge'], self.slots['k'], self.slots['lane'])):
            e, k = int(e0), int(k0)
            es = [e] if lane0 is None or int(lane0) < 0 else list(map(int, self.lane_edges[int(lane0)]))
            if not es:
                es = [e]
            self.routes.append((i, int(self.tail[es[0]]), int(self.head[es[-1]]), k, es))
        self.lane_dest = {}
        for lane, es0 in enumerate(self.lane_edges):
            es = list(map(int, es0))
            if es:
                for e in es:
                    self.lane_dest[(e, lane)] = int(self.head[es[-1]])
        self.nominal_caps = []
        for _, _, _, _, es in self.routes:
            vals = [float(self.edges['u0'][e]) for e in es if self.edges['u0'][e] is not None]
            self.nominal_caps.append(min(vals) if vals else 1e9)

    def _dest(self, edge, lane):
        return self.lane_dest.get((int(edge), int(lane)), int(self.head[int(edge)]))

    def _fallback(self, o):
        flows = np.zeros(self.nslot, dtype=float)
        u = np.asarray(o['graph_now.u'], dtype=float)
        src_qty = {}
        for key, row in self.stock_rows.items():
            src_qty[key] = max(0.0, float(o['stock.qty'][row]))
        for key, row in self.supply_rows.items():
            src_qty[key] = src_qty.get(key, 0.0) + max(0.0, float(o['graph_now.supply.avail'][row]))
        demand = {}
        forecast = np.asarray(o['demand_forecast.qty'], dtype=float)
        fm = o.get('demand_forecast.qty.observed')
        for key, row in self.demand_rows.items():
            vals = forecast[row]
            if fm is not None:
                vals = vals[np.asarray(fm[row], dtype=bool)]
            demand[key] = (max(0.0, float(np.mean(vals))) if len(vals) else 0.0) + max(0.0, float(o['backlog.qty'][row]))
        edge_left = np.maximum(0.0, u.copy())
        candidates = []
        mask = np.asarray(o['action_mask'], dtype=bool)
        c = np.asarray(o['graph_now.c'], dtype=float)
        tariff = np.asarray(o['graph_now.tariff'], dtype=float)
        for i, src, dst, k, es in self.routes:
            if not mask[i] or (dst, k) not in demand:
                continue
            cap = min([self.nominal_caps[i]] + [max(0.0, float(u[e])) for e in es])
            if cap <= 0:
                continue
            cost = sum(float(c[e]) + float(tariff[e, k]) * self.v[k] for e in es)
            candidates.append((cost, i, src, dst, k, es, cap))
        for _, i, src, dst, k, es, cap in sorted(candidates):
            need = max(0.0, demand[(dst, k)] - src_qty.get((dst, k), 0.0))
            qty = min(need, cap, src_qty.get((src, k), 0.0))
            for e in es:
                qty = min(qty, edge_left[e])
            if qty <= 0:
                continue
            flows[i] = qty
            src_qty[(src, k)] -= qty
            src_qty[(dst, k)] = src_qty.get((dst, k), 0.0) + qty
            demand[(dst, k)] -= qty
            for e in es:
                edge_left[e] -= qty
        return {'flows': flows}

    def act(self, o):
        if linprog is None:
            return self._fallback(o)
        week = int(o['week'][0])
        remaining = max(1, self.T - week + 1)
        max_tau = max([int(x or 0) for x in self.edges.get('tau0', [])] + [1])
        H = min(remaining, max(2, min(6, max_tau + 2)))
        nstate = self.nnode * self.nk
        u = np.asarray(o['graph_now.u'], dtype=float)
        c = np.asarray(o['graph_now.c'], dtype=float)
        tau = np.asarray(o['graph_now.tau'], dtype=float)
        tariff = np.asarray(o['graph_now.tariff'], dtype=float)
        mask = np.asarray(o['action_mask'], dtype=bool)
        opens = np.asarray(o['graph_now.open'], dtype=float)

        # Route variables are indexed by dispatch offset, then action slot.
        rv = []
        for h in range(H):
            for i, src, dst, k, es in self.routes:
                if h == 0 and not mask[i]:
                    continue
                lead_raw = sum(max(0, int(tau[e])) if h == 0 else max(0, int(self.edges['tau0'][e] or 0)) for e in es)
                lead = max(1, lead_raw)
                arrival = h + lead
                if arrival >= H:
                    continue
                cap = min([self.nominal_caps[i]] + [max(0.0, float(u[e]) if h == 0 else float(self.edges['u0'][e] or 0.0)) for e in es])
                if h == 0:
                    for e in es:
                        cp = self.cp_rows.get(int(self.head[e]))
                        if cp is not None:
                            cap *= max(0.0, float(opens[cp]))
                if cap <= 1e-9:
                    continue
                cost = 0.0
                for e in es:
                    freight = float(c[e]) if h == 0 else float(self.edges['c0'][e] or 0.0)
                    tr = float(tariff[e, k]) if h == 0 else 0.0
                    cost += freight + tr * self.v[k]
                cost += 0.005 * self.v[k] * lead
                rv.append((h, i, src, dst, k, es, arrival, cap, cost))
        if not rv:
            return {'flows': np.zeros(self.nslot, dtype=float)}

        sink_vars = [(h, d, int(node), int(k)) for h in range(H) for d, (node, k) in enumerate(zip(self.sink_nodes, self.sink_ks))]
        nr, ns = len(rv), len(sink_vars)
        demand_amt = np.zeros(ns, dtype=float)
        for z, (h, d, node, k) in enumerate(sink_vars):
            row = self.demand_rows.get((node, k))
            if row is None:
                continue
            f = np.asarray(o['demand_forecast.qty'][row], dtype=float)
            fm = o.get('demand_forecast.qty.observed')
            if h < len(f) and (fm is None or bool(fm[row, h])):
                dem = max(0.0, float(f[h]))
            else:
                vals = f if fm is None else f[np.asarray(fm[row], dtype=bool)]
                dem = max(0.0, float(np.mean(vals))) if len(vals) else 0.0
            if h == 0:
                dem += max(0.0, float(o['backlog.qty'][row]))
            demand_amt[z] = dem

        # Variables: route shipments, unmet demand, and end-of-period inventory.
        inv0 = nr + ns
        inv_idx = np.arange(inv0, inv0 + H * nstate).reshape(H, nstate)
        nv = inv0 + H * nstate
        obj = np.zeros(nv, dtype=float)
        for j, x in enumerate(rv):
            obj[j] = x[8] + 1e-7
        for z, (_, d, _, _) in enumerate(sink_vars):
            pi = float(self.sink_pi[d]) if d < len(self.sink_pi) else float(np.max(self.v))
            obj[nr + z] = -max(1.0, pi)
        obj[inv0:] = 1e-5

        initial = np.zeros((H, nstate), dtype=float)
        for (node, k), row in self.stock_rows.items():
            initial[0, node * self.nk + k] += max(0.0, float(o['stock.qty'][row]))
        for (node, k), row in self.supply_rows.items():
            initial[0, node * self.nk + k] += max(0.0, float(o['graph_now.supply.avail'][row]))
        pq = np.asarray(o.get('pipeline.qty', []), dtype=float)
        pm = np.asarray(o.get('pipeline.qty.observed', np.ones_like(pq)), dtype=bool)
        plm = np.asarray(o.get('pipeline.lane.observed', np.ones_like(pm)), dtype=bool)
        for j in np.flatnonzero(pm):
            q = max(0.0, float(pq[j]))
            if q <= 0:
                continue
            off = int(o['pipeline.arrival_week'][j]) - week
            if 0 <= off < H:
                e, k = int(o['pipeline.edge'][j]), int(o['pipeline.k'][j])
                lane = int(o['pipeline.lane'][j]) if plm[j] else -1
                node = self._dest(e, lane)
                initial[off, node * self.nk + k] += q
        # Lots in a queue are treated as a near-term receipt at their lane's destination.
        lots = o.get('queue_lots.qty')
        if lots is not None:
            for r, key in enumerate(self.lot_keys):
                if r >= len(lots):
                    break
                cp, k, lane, edge = map(int, key)
                q = max(0.0, float(np.sum(lots[r])))
                if q:
                    node = self._dest(edge, lane)
                    initial[0, node * self.nk + k] += q

        eq, beq = [], []
        for h in range(H):
            for st in range(nstate):
                row = np.zeros(nv, dtype=float)
                row[inv_idx[h, st]] = 1.0
                if h:
                    row[inv_idx[h - 1, st]] = -1.0
                b = initial[h, st]
                for j, x in enumerate(rv):
                    rh, _, src, dst, k, _, arrival, _, _ = x
                    if rh == h and src * self.nk + k == st:
                        row[j] += 1.0
                    if arrival == h and dst * self.nk + k == st:
                        row[j] -= 1.0
                for z, (sh, _, node, k) in enumerate(sink_vars):
                    if sh == h and node * self.nk + k == st:
                        row[nr + z] += 1.0
                        b -= demand_amt[z]
                eq.append(row)
                beq.append(b)

        aub, bub = [], []
        # Shared edge capacities across all route slots using an edge.
        for h in range(H):
            for e in range(len(self.tail)):
                ids = [j for j, x in enumerate(rv) if x[0] == h and e in x[5]]
                if ids:
                    cap = max(0.0, float(u[e]) if h == 0 else float(self.edges['u0'][e] or 0.0))
                    row = np.zeros(nv, dtype=float)
                    row[ids] = 1.0
                    aub.append(row)
                    bub.append(cap)
        bounds = [(0.0, x[7]) for x in rv] + [(0.0, demand_amt[z]) for z in range(ns)] + [(0.0, None)] * (H * nstate)
        try:
            res = linprog(obj, A_ub=np.asarray(aub) if aub else None, b_ub=np.asarray(bub) if bub else None,
                          A_eq=np.asarray(eq), b_eq=np.asarray(beq), bounds=bounds, method='highs')
        except Exception:
            return self._fallback(o)
        if not res.success or res.x is None:
            return self._fallback(o)
        flows = np.zeros(self.nslot, dtype=float)
        for j, x in enumerate(rv):
            if x[0] == 0:
                flows[x[1]] += max(0.0, float(res.x[j]))
        return {'flows': flows}
