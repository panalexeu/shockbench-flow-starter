# -2.107824162609576
import numpy as np


class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.T = int(config['T'])
        e = self.s['edges']
        self.tail = np.asarray(e['tail'], dtype=int)
        self.head = np.asarray(e['head'], dtype=int)
        self.u0 = np.asarray([0.0 if x is None else float(x) for x in e['u0']])
        self.c0 = np.asarray([0.0 if x is None else float(x) for x in e['c0']])
        self.tau0 = np.asarray([1.0 if x is None else float(x) for x in e['tau0']])
        self.values = np.asarray(self.s['commodities']['v'], dtype=float)
        self.pools = self.s['commodities'].get('pool', ['tb'] * len(self.values))
        self.stocks = [tuple(map(int, x)) for x in self.l['stock_slots']]
        self.supplies = [tuple(map(int, x)) for x in self.l['supply_slots']]
        self.demands = [tuple(map(int, x)) for x in self.l['demands']]
        self.demand_index = {p: i for i, p in enumerate(self.demands)}
        self.cp_index = {int(n): i for i, n in enumerate(self.l['chokepoints'])}
        self.sink_info = {}
        for n, k, pi, carry in zip(self.s['sinks']['node'], self.s['sinks']['k'],
                                   self.s['sinks']['pi'], self.s['sinks']['backlog']):
            self.sink_info[(int(n), int(k))] = (float(pi), bool(carry))
        self.nnode = len(self.s['nodes']['id'])
        self.routes = []
        slots = self.s['action_slots']
        for i, (edge, k, lane) in enumerate(zip(slots['edge'], slots['k'], slots['lane'])):
            edge, k = int(edge), int(k)
            lane = -1 if lane is None else int(lane)
            es = [edge] if lane < 0 else [int(x) for x in self.s['lanes']['edges'][lane]]
            cps = [] if lane < 0 else [int(x) for x in self.s['lanes']['chokepoints'][lane]]
            if not es:
                continue
            self.routes.append((i, k, es, cps, int(self.tail[es[0]]), int(self.head[es[-1]])))
        self.nslot = len(slots['edge'])
        self.lane_edges = self.s['lanes']['edges']

    @staticmethod
    def field(o, name, fallback):
        fb = np.asarray(fallback)
        if name not in o:
            return fb.copy()
        x = np.asarray(o[name])
        m = o.get(name + '.observed')
        if m is not None and np.shape(m) == x.shape:
            return np.where(np.asarray(m, dtype=bool), x, fb)
        return x.copy()

    def act(self, o):
        week = int(np.asarray(o['week']).reshape(-1)[0])
        ne, nk = len(self.tail), len(self.values)
        u = self.field(o, 'graph_now.u', self.u0).astype(float)
        c = self.field(o, 'graph_now.c', self.c0).astype(float)
        tau = np.maximum(1.0, self.field(o, 'graph_now.tau', self.tau0).astype(float))
        tariff = self.field(o, 'graph_now.tariff', np.zeros((ne, nk))).astype(float)
        banned = self.field(o, 'graph_now.prohibited', np.zeros((ne, nk))).astype(bool)
        opened = self.field(o, 'graph_now.open', np.ones(len(self.cp_index))).astype(float)
        mask = np.asarray(o.get('action_mask', np.ones(self.nslot)), dtype=bool).reshape(-1)

        closure_end = {}
        ce = np.asarray(o.get('closure_end.chokepoint', [])).reshape(-1)
        ew = np.asarray(o.get('closure_end.end_week', np.zeros(ce.size))).reshape(-1)
        cem = np.asarray(o.get('closure_end.chokepoint.observed', np.ones(ce.size)), dtype=bool).reshape(-1)
        ewm = np.asarray(o.get('closure_end.end_week.observed', np.zeros(ce.size)), dtype=bool).reshape(-1)
        for j in range(min(ce.size, ew.size, cem.size, ewm.size)):
            if cem[j] and ewm[j]:
                cp = int(ce[j])
                closure_end[cp] = max(closure_end.get(cp, week), int(ew[j]) + 1)

        pending = {}
        pe = np.asarray(o.get('pending_prohibitions.edge', [])).reshape(-1)
        pk = np.asarray(o.get('pending_prohibitions.k', np.zeros(pe.size))).reshape(-1)
        pw = np.asarray(o.get('pending_prohibitions.effective_week', np.full(pe.size, self.T + 1))).reshape(-1)
        pem = np.asarray(o.get('pending_prohibitions.edge.observed', np.ones(pe.size)), dtype=bool).reshape(-1)
        for j in range(min(pe.size, pk.size, pw.size, pem.size)):
            if pem[j]:
                key = (int(pe[j]), int(pk[j]))
                pending[key] = min(pending.get(key, self.T + 1), int(pw[j]))

        warning = self.field(o, 'warning.score', np.zeros(len(self.l.get('warning_units', [])))).reshape(-1)
        warning_cp = {}
        for unit, score in zip(self.l.get('warning_units', []), warning):
            if len(unit) >= 2 and unit[0] == 'chokepoint':
                warning_cp[int(unit[1])] = float(np.clip(score, 0.0, 1.0))

        def route_legal(route):
            i, k, es, cps, src, dst = route
            if i >= mask.size or not mask[i] or any(banned[e, k] for e in es):
                return False
            if any(cp in self.cp_index and opened[self.cp_index[cp]] <= 0.001 for cp in cps):
                return False
            elapsed = 0.0
            for edge in es:
                if pending.get((edge, k), self.T + 1) <= week + elapsed:
                    return False
                elapsed += tau[edge]
            return True

        # Estimate sink-directed demand rates for intermediate-node replenishment.
        forecast = np.asarray(o.get('demand_forecast.qty', np.zeros((len(self.demands), 1))), dtype=float)
        backlog = self.field(o, 'backlog.qty', np.zeros(len(self.demands))).astype(float).reshape(-1)
        demand_rate = {}
        for p, di in self.demand_index.items():
            if forecast.ndim == 2 and di < forecast.shape[0] and forecast.shape[1]:
                row = forecast[di]
                rate = float(np.mean(row[:min(4, len(row))]))
            else:
                rate = 0.0
            demand_rate[p] = max(0.0, rate)
        for _ in range(self.nnode):
            changed = False
            for r in self.routes:
                _, k, _, _, src, dst = r
                value = demand_rate.get((dst, k), 0.0)
                if value > demand_rate.get((src, k), 0.0) + 1e-9:
                    demand_rate[(src, k)] = value
                    changed = True
            if not changed:
                break

        # Build a downstream cost-to-sink label. Each node inherits the best
        # (route cost / shortage penalty) continuation for its commodity.
        route_cost = {}
        for r in self.routes:
            i, k, es, cps, src, dst = r
            if not route_legal(r):
                continue
            freight = sum(float(c[e]) + float(tariff[e, k]) * float(self.values[k]) for e in es)
            risk_scale = max(1.0, 0.05 * max(0.0, freight))
            risk = sum(warning_cp.get(cp, 0.0) for cp in cps)
            route_cost[i] = freight + risk_scale * risk
        label_cost, label_pi, label_ratio = {}, {}, {}
        for (node, k), (pi, _) in self.sink_info.items():
            if pi > 0:
                label_cost[(node, k)] = 0.0
                label_pi[(node, k)] = pi
                label_ratio[(node, k)] = 0.0
        for _ in range(self.nnode):
            changed = False
            for r in self.routes:
                i, k, es, cps, src, dst = r
                if i not in route_cost:
                    continue
                key_dst, key_src = (dst, k), (src, k)
                if key_dst not in label_cost:
                    continue
                pi = label_pi[key_dst]
                cost = route_cost[i] + label_cost[key_dst]
                ratio = cost / max(pi, 1e-9)
                if ratio + 1e-10 < label_ratio.get(key_src, np.inf):
                    label_cost[key_src] = cost
                    label_pi[key_src] = pi
                    label_ratio[key_src] = ratio
                    changed = True
            if not changed:
                break

        # Current dispatchable inventory and projected arrivals at each node.
        stock = self.field(o, 'stock.qty', np.zeros(len(self.stocks))).astype(float).reshape(-1)
        available = {}
        projected = {}

        def add(p, arrival, qty):
            if qty > 0 and np.isfinite(qty):
                projected.setdefault(p, []).append((int(arrival), float(qty)))

        for j, p in enumerate(self.stocks):
            q = max(0.0, float(stock[j])) if j < stock.size else 0.0
            available[p] = q
            add(p, week, q)
        supply = self.field(o, 'graph_now.supply.avail', np.zeros(len(self.supplies))).astype(float).reshape(-1)
        for j, p in enumerate(self.supplies):
            q = max(0.0, float(supply[j])) if j < supply.size else 0.0
            available[p] = available.get(p, 0.0) + q

        def lane_info(edge, lane):
            if lane < 0 or lane >= len(self.lane_edges):
                return [edge]
            return [int(x) for x in self.lane_edges[lane]]

        pq = np.asarray(o.get('pipeline.qty', [])).reshape(-1)
        plive = np.asarray(o.get('pipeline.qty.observed', np.ones(pq.size)), dtype=bool).reshape(-1)
        pedge = np.asarray(o.get('pipeline.edge', np.zeros(pq.size))).reshape(-1)
        pk = np.asarray(o.get('pipeline.k', np.zeros(pq.size))).reshape(-1)
        plane = np.asarray(o.get('pipeline.lane', np.full(pq.size, -1))).reshape(-1)
        planem = np.asarray(o.get('pipeline.lane.observed', np.ones(pq.size)), dtype=bool).reshape(-1)
        parr = np.asarray(o.get('pipeline.arrival_week', np.full(pq.size, week))).reshape(-1)
        for j in range(min(pq.size, pedge.size, pk.size, parr.size, plive.size)):
            edge = int(pedge[j])
            if not plive[j] or pq[j] <= 0 or edge < 0 or edge >= ne:
                continue
            lane = int(plane[j]) if j < plane.size and j < planem.size and planem[j] else -1
            es = lane_info(edge, lane)
            try:
                pos = es.index(edge)
            except ValueError:
                pos = 0
            dst = int(self.head[es[-1]])
            remaining = sum(tau[x] for x in es[pos + 1:] if 0 <= x < ne)
            arrival = max(week, int(parr[j])) + int(np.ceil(remaining))
            add((dst, int(pk[j])), arrival, pq[j])

        wq = np.asarray(o.get('wip.qty', [])).reshape(-1)
        wlive = np.asarray(o.get('wip.qty.observed', np.ones(wq.size)), dtype=bool).reshape(-1)
        wn = np.asarray(o.get('wip.node', np.zeros(wq.size))).reshape(-1)
        wk = np.asarray(o.get('wip.k', np.zeros(wq.size))).reshape(-1)
        wo = np.asarray(o.get('wip.out_week', np.full(wq.size, week))).reshape(-1)
        for j in range(min(wq.size, wlive.size, wn.size, wk.size, wo.size)):
            if wlive[j] and wq[j] > 0:
                add((int(wn[j]), int(wk[j])), max(week, int(wo[j])), wq[j])

        lots = np.asarray(o.get('queue_lots.qty', np.zeros((0, 0))))
        lot_mask = np.asarray(o.get('queue_lots.qty.observed', np.ones(lots.shape)), dtype=bool)
        for row, key in enumerate(self.l.get('lot_keys', [])[:lots.shape[0]]):
            if len(key) < 4:
                continue
            cp, k, lane, edge = key
            cp, k, edge = int(cp), int(k), int(edge)
            lane = -1 if lane is None else int(lane)
            if edge < 0 or edge >= ne:
                continue
            release = week
            ci = self.cp_index.get(cp)
            if ci is not None and opened[ci] <= 0.001:
                release = max(week, closure_end.get(cp, week + 4))
            es = lane_info(edge, lane)
            try:
                pos = es.index(edge)
            except ValueError:
                pos = 0
            dst = int(self.head[es[-1]])
            remaining = sum(tau[x] for x in es[pos:] if 0 <= x < ne)
            if row < lots.shape[0]:
                valid = lot_mask[row] if row < lot_mask.shape[0] else np.ones(lots.shape[1], dtype=bool)
                qty = float(np.sum(np.where(valid, lots[row], 0.0)))
                add((dst, k), release + int(np.ceil(remaining)), qty)

        def coverage(p, cutoff):
            return sum(q for arrival, q in projected.get(p, []) if arrival <= cutoff)

        def need(dst, k, lead):
            p = (dst, k)
            remaining_weeks = max(1, self.T - week + 1)
            if p in self.demand_index:
                di = self.demand_index[p]
                h = min(remaining_weeks, max(1, int(np.ceil(lead)) + 2))
                total = max(0.0, float(backlog[di])) if di < backlog.size else 0.0
                if forecast.ndim == 2 and di < forecast.shape[0] and forecast.shape[1]:
                    row = forecast[di]
                    n = min(h, len(row))
                    total += float(np.sum(row[:n]))
                    if h > n and n:
                        total += (h - n) * float(np.mean(row[-min(3, n):]))
                return max(0.0, total - coverage(p, week + h - 1))
            duration = min(max(2.0, lead + 2.0), float(remaining_weeks))
            target = max(0.0, demand_rate.get(p, 0.0)) * duration
            return max(0.0, target - coverage(p, week + int(np.ceil(duration))))

        edge_left = np.maximum(0.0, u.copy())
        cp_left = {}
        for pool in ('tb', 'ct'):
            caps = self.field(o, 'graph_now.kappa.' + pool,
                              np.full(len(self.cp_index), np.inf)).astype(float).reshape(-1)
            for cp, ci in self.cp_index.items():
                cap = float(caps[ci]) if ci < caps.size else np.inf
                cp_left[(cp, pool)] = max(0.0, cap * max(0.0, float(opened[ci])))

        candidates = []
        for r in self.routes:
            i, k, es, cps, src, dst = r
            if i not in route_cost or not route_legal(r):
                continue
            lead = sum(tau[e] for e in es)
            if week + lead > self.T + 1 or min((u[e] for e in es), default=0.0) <= 0:
                continue
            if need(dst, k, lead) <= 0:
                continue
            path_cost = route_cost[i]
            downstream = label_cost.get((dst, k), 0.0)
            pi = label_pi.get((dst, k), 0.0)
            if pi > 0:
                score = (path_cost + downstream + 0.002 * self.values[k] * lead) / pi
            else:
                score = path_cost + 0.002 * self.values[k] * lead + 1e3
            candidates.append((score, lead, i, r))

        flows = np.zeros(self.nslot, dtype=float)
        for _, lead, i, r in sorted(candidates, key=lambda x: (x[0], x[1], x[2])):
            _, k, es, cps, src, dst = r
            q = min(available.get((src, k), 0.0), need(dst, k, lead))
            for edge in es:
                q = min(q, max(0.0, edge_left[edge]))
            for cp in cps:
                if cp in self.cp_index:
                    q = min(q, cp_left.get((cp, self.pools[k]), 0.0))
            if not np.isfinite(q) or q <= 0:
                continue
            flows[i] = q
            available[(src, k)] = max(0.0, available.get((src, k), 0.0) - q)
            for edge in es:
                edge_left[edge] = max(0.0, edge_left[edge] - q)
            for cp in cps:
                if cp in self.cp_index:
                    key = (cp, self.pools[k])
                    cp_left[key] = max(0.0, cp_left.get(key, 0.0) - q)
            add((dst, k), week + int(np.ceil(lead)), q)

        return {'flows': flows}
