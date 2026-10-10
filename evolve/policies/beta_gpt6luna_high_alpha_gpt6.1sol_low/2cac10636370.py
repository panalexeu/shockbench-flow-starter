# -1.634267781576407
import numpy as np


class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.T = int(config['T'])
        s = self.s
        e = s['edges']
        self.tail = np.asarray(e['tail'], dtype=int)
        self.head = np.asarray(e['head'], dtype=int)
        self.u0 = np.asarray([0.0 if x is None else float(x) for x in e['u0']])
        self.c0 = np.asarray([0.0 if x is None else float(x) for x in e['c0']])
        self.tau0 = np.asarray([1.0 if x is None else float(x) for x in e['tau0']])
        self.values = np.asarray(s['commodities']['v'], dtype=float)
        self.pools = s['commodities'].get('pool', ['tb'] * len(self.values))
        self.stock_slots = [tuple(map(int, p)) for p in self.l['stock_slots']]
        self.supply_slots = [tuple(map(int, p)) for p in self.l['supply_slots']]
        self.demands = [tuple(map(int, p)) for p in self.l['demands']]
        self.demand_index = {p: i for i, p in enumerate(self.demands)}
        self.cp_index = {int(n): i for i, n in enumerate(self.l['chokepoints'])}
        self.nslots = len(s['action_slots']['edge'])
        self.node_types = s['nodes'].get('type', [])
        self.node_regions = s['nodes'].get('region', [])
        self.routes = []
        grouped = {}
        slots = s['action_slots']
        for i, (edge, k, lane) in enumerate(zip(slots['edge'], slots['k'], slots['lane'])):
            edge, k = int(edge), int(k)
            lane = -1 if lane is None else int(lane)
            es = [edge] if lane < 0 else [int(x) for x in s['lanes']['edges'][lane]]
            cps = [] if lane < 0 else [int(x) for x in s['lanes']['chokepoints'][lane]]
            if not es:
                continue
            src, dst = int(self.tail[es[0]]), int(self.head[es[-1]])
            self.routes.append((i, k, es, cps, src, dst))
            alt = e['alt_of'][edge] if lane < 0 else s['lanes']['alt_of'][lane]
            cap = min(self.u0[x] for x in es)
            grouped.setdefault((src, k, dst), []).append((alt, cap))

        incoming, outgoing = {}, {}
        for (src, k, dst), choices in grouped.items():
            base = [cap for alt, cap in choices if alt is None]
            cap = sum(base) if base else max(x[1] for x in choices)
            incoming[(dst, k)] = incoming.get((dst, k), 0.0) + cap
            outgoing[(src, k)] = outgoing.get((src, k), 0.0) + cap
        self.node_rate = {}
        for p in set(incoming) | set(outgoing):
            a, b = incoming.get(p), outgoing.get(p)
            self.node_rate[p] = min(a, b) if a is not None and b is not None else max(a or 0.0, b or 0.0)

        sink_pi = {}
        for n, k, pi in zip(s['sinks']['node'], s['sinks']['k'], s['sinks']['pi']):
            sink_pi[(int(n), int(k))] = max(1.0, float(pi))
        self.pi = np.asarray([sink_pi.get(p, 1.0) for p in self.demands])
        self.max_pi = max(1.0, float(self.pi.max()) if self.pi.size else 1.0)

        reverse = {}
        for _, k, _, _, src, dst in self.routes:
            reverse.setdefault((dst, k), set()).add(src)
        self.priority = {}
        for di, (sink, k) in enumerate(self.demands):
            queue, seen = [(sink, 0)], {sink}
            for node, dist in queue:
                p = (node, k)
                self.priority[p] = max(self.priority.get(p, 0.0), self.pi[di] / (1.0 + 0.12 * dist))
                for prev in reverse.get(p, ()):
                    if prev not in seen:
                        seen.add(prev)
                        queue.append((prev, dist + 1))

    @staticmethod
    def field(o, name, fallback):
        fb = np.asarray(fallback)
        if name not in o:
            return fb.copy()
        x = np.asarray(o[name])
        mask = o.get(name + '.observed')
        if mask is not None and np.shape(mask) == x.shape:
            return np.where(mask, x, fb)
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
        mask = np.asarray(o.get('action_mask', np.ones(self.nslots)), dtype=bool).reshape(-1)
        forecast = np.asarray(o.get('demand_forecast.qty', np.zeros((len(self.demands), 1))), dtype=float)
        backlog = np.asarray(o.get('backlog.qty', np.zeros(len(self.demands))), dtype=float).reshape(-1)

        stock_left = {}
        available = {}
        for i, p in enumerate(self.stock_slots):
            q = max(0.0, float(self.field(o, 'stock.qty', np.zeros(len(self.stock_slots)))[i]))
            stock_left[p] = stock_left.get(p, 0.0) + q
            available[p] = available.get(p, 0.0) + q
        supply = self.field(o, 'graph_now.supply.avail', np.zeros(len(self.supply_slots)))
        for p, q in zip(self.supply_slots, supply):
            available[p] = available.get(p, 0.0) + max(0.0, float(q))

        projected = {}

        def add_projection(pair, arrival, qty):
            if qty > 0:
                projected.setdefault(pair, []).append((int(arrival), float(qty)))

        lanes = self.s['lanes']['edges']
        pq = np.asarray(o.get('pipeline.qty', [])).reshape(-1)
        plive = np.asarray(o.get('pipeline.qty.observed', np.ones(pq.size)), dtype=bool).reshape(-1)
        pe = np.asarray(o.get('pipeline.edge', np.zeros(pq.size)), dtype=int).reshape(-1)
        pk = np.asarray(o.get('pipeline.k', np.zeros(pq.size)), dtype=int).reshape(-1)
        plane = np.asarray(o.get('pipeline.lane', np.full(pq.size, -1)), dtype=int).reshape(-1)
        plm = np.asarray(o.get('pipeline.lane.observed', np.ones(pq.size)), dtype=bool).reshape(-1)
        parr = np.asarray(o.get('pipeline.arrival_week', np.full(pq.size, week)), dtype=int).reshape(-1)
        for j in range(min(pq.size, pe.size, pk.size, parr.size)):
            if j >= plive.size or not plive[j] or pq[j] <= 0 or not 0 <= pe[j] < ne:
                continue
            lane = int(plane[j]) if j < plane.size and j < plm.size and plm[j] else -1
            dst, arrival = int(self.head[pe[j]]), int(parr[j])
            if 0 <= lane < len(lanes):
                es = [int(x) for x in lanes[lane]]
                dst = int(self.head[es[-1]])
                if pe[j] in es:
                    pos = es.index(pe[j])
                    arrival += int(np.ceil(sum(tau[e] for e in es[pos + 1:])))
            add_projection((dst, int(pk[j])), max(week, arrival), pq[j])

        wq = np.asarray(o.get('wip.qty', [])).reshape(-1)
        wlive = np.asarray(o.get('wip.qty.observed', np.ones(wq.size)), dtype=bool).reshape(-1)
        wn = np.asarray(o.get('wip.node', np.zeros(wq.size)), dtype=int).reshape(-1)
        wk = np.asarray(o.get('wip.k', np.zeros(wq.size)), dtype=int).reshape(-1)
        wo = np.asarray(o.get('wip.out_week', np.full(wq.size, week)), dtype=int).reshape(-1)
        for j in range(min(wq.size, wn.size, wk.size, wo.size)):
            if j < wlive.size and wlive[j] and wq[j] > 0:
                add_projection((int(wn[j]), int(wk[j])), max(week, int(wo[j])), wq[j])

        closure_end = {}
        ce = np.asarray(o.get('closure_end.chokepoint', []), dtype=int).reshape(-1)
        cem = np.asarray(o.get('closure_end.chokepoint.observed', np.zeros(ce.size)), dtype=bool).reshape(-1)
        ew = np.asarray(o.get('closure_end.end_week', np.zeros(ce.size)), dtype=int).reshape(-1)
        ewm = np.asarray(o.get('closure_end.end_week.observed', np.zeros(ce.size)), dtype=bool).reshape(-1)
        for j in range(min(ce.size, cem.size, ew.size, ewm.size)):
            if cem[j] and ewm[j]:
                closure_end[int(ce[j])] = int(ew[j]) + 1
        lots = np.asarray(o.get('queue_lots.qty', np.zeros((0, 0))), dtype=float)
        lot_live = np.asarray(o.get('queue_lots.qty.observed', np.ones(lots.shape)), dtype=bool)
        for row, key in enumerate(self.l.get('lot_keys', [])[:lots.shape[0]]):
            if len(key) < 4:
                continue
            cp, k, lane, edge = key
            edge = int(edge)
            if not 0 <= edge < ne or row >= lot_live.shape[0]:
                continue
            lane = -1 if lane is None else int(lane)
            es = [int(x) for x in lanes[lane]] if 0 <= lane < len(lanes) else [edge]
            dst = int(self.head[es[-1]])
            start = max(week, closure_end.get(int(cp), week))
            ci = self.cp_index.get(int(cp))
            if ci is not None and opened[ci] <= 0.001 and int(cp) not in closure_end:
                start += 3
            rem = sum(tau[e] for e in es[es.index(edge):]) if edge in es else tau[edge]
            qty = float(np.sum(np.where(lot_live[row], lots[row], 0.0)))
            add_projection((dst, int(k)), start + int(np.ceil(rem)), qty)

        pending = {}
        p_edge = np.asarray(o.get('pending_prohibitions.edge', []), dtype=int).reshape(-1)
        p_k = np.asarray(o.get('pending_prohibitions.k', np.zeros(p_edge.size)), dtype=int).reshape(-1)
        p_week = np.asarray(o.get('pending_prohibitions.effective_week', np.full(p_edge.size, self.T + 1)), dtype=int).reshape(-1)
        p_live = np.asarray(o.get('pending_prohibitions.edge.observed', np.ones(p_edge.size)), dtype=bool).reshape(-1)
        for j in range(min(p_edge.size, p_k.size, p_week.size)):
            if j < p_live.size and p_live[j]:
                key = (int(p_edge[j]), int(p_k[j]))
                pending[key] = min(pending.get(key, self.T + 1), int(p_week[j]))

        def coverage(pair, cutoff):
            return stock_left.get(pair, 0.0) + sum(q for arrival, q in projected.get(pair, ()) if arrival <= cutoff)

        def requirement(pair, lead):
            remaining = max(1, self.T - week + 1)
            if pair in self.demand_index:
                di = self.demand_index[pair]
                h = min(remaining, max(1, int(np.ceil(lead)) + 2))
                row = forecast[di] if forecast.ndim == 2 and di < forecast.shape[0] else np.zeros(0)
                n = min(h, row.size)
                total = max(0.0, float(backlog[di])) if di < backlog.size else 0.0
                if n:
                    total += float(np.sum(row[:n]))
                if h > n and row.size:
                    total += (h - n) * max(0.0, float(np.mean(row[-min(3, row.size):])))
                cutoff = week + h - 1
            else:
                duration = min(max(1.0, lead + 2.0), float(remaining))
                total = max(0.0, self.node_rate.get(pair, 0.0)) * duration
                cutoff = week + int(np.ceil(duration))
            gap = max(0.0, total - coverage(pair, cutoff))
            return gap, total, cutoff

        cp_left = {}
        cp_caps = {}
        for pool in ('tb', 'ct'):
            cp_caps[pool] = self.field(o, 'graph_now.kappa.' + pool,
                                       np.full(len(self.cp_index), np.inf)).astype(float)
            for cp, ci in self.cp_index.items():
                cap = cp_caps[pool][ci] if ci < cp_caps[pool].size else np.inf
                cp_left[(cp, pool)] = max(0.0, float(cap) * max(0.0, float(opened[ci])))
        edge_left = np.maximum(0.0, u.copy())

        warning = self.field(o, 'warning.score', np.zeros(len(self.l.get('warning_units', [])))).reshape(-1)
        warning_map = {tuple(key): float(np.clip(warning[i], 0.0, 1.0))
                       for i, key in enumerate(self.l.get('warning_units', [])) if i < warning.size}
        candidates = []
        for route in self.routes:
            i, k, es, cps, src, dst = route
            if i >= mask.size or not mask[i] or any(banned[e, k] for e in es):
                continue
            lead = float(sum(tau[e] for e in es))
            if week + lead > self.T + 1:
                continue
            elapsed = 0.0
            risky = False
            for e in es:
                if pending.get((e, k), self.T + 1) <= week + elapsed:
                    risky = True
                    break
                elapsed += tau[e]
            if risky:
                continue
            cap = min((max(0.0, float(u[e])) for e in es), default=0.0)
            for cp in cps:
                cap = min(cap, cp_left.get((cp, self.pools[k]), 0.0))
            if cap <= 0:
                continue
            pair = (dst, k)
            node_type = self.node_types[dst] if dst < len(self.node_types) else ''
            # Serve a sink or replenish a node that can use or forward this good.
            if pair not in self.demand_index and pair not in self.priority and node_type not in ('fab', 'osat', 'grid'):
                continue
            gap, total, _ = requirement(pair, lead)
            if gap <= 0:
                continue
            priority = self.priority.get(pair, self.max_pi * 0.12)
            if pair in self.demand_index:
                priority = max(1.0, priority)
            else:
                priority = max(1.0, priority)
            freight = sum(float(c[e]) + float(tariff[e, k]) * float(self.values[k]) for e in es)
            risk = max([warning_map.get(('chokepoint', cp), 0.0) for cp in cps] +
                       [warning_map.get(('region', int(self.node_regions[n])), 0.0)
                        for n in (src, dst) if n < len(self.node_regions)] + [0.0])
            cost = max(0.0, freight) + 0.002 * float(self.values[k]) * lead
            cost *= 1.0 + 0.20 * risk
            urgency = 0.35 + 0.65 * min(1.0, gap / max(total, 1e-9))
            candidates.append((cost / (priority * urgency), lead, i, cap, route))

        flows = np.zeros(self.nslots, dtype=float)
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        for _, lead, i, planned_cap, route in candidates:
            _, k, es, cps, src, dst = route
            pair = (dst, k)
            gap, _, _ = requirement(pair, lead)
            q = min(planned_cap, gap, max(0.0, available.get((src, k), 0.0)))
            for e in es:
                q = min(q, max(0.0, edge_left[e]))
            for cp in cps:
                q = min(q, cp_left.get((cp, self.pools[k]), 0.0))
            if q <= 1e-9 or not np.isfinite(q):
                continue
            flows[i] += q
            available[(src, k)] = max(0.0, available.get((src, k), 0.0) - q)
            used_stock = min(q, stock_left.get((src, k), 0.0))
            stock_left[(src, k)] = max(0.0, stock_left.get((src, k), 0.0) - used_stock)
            for e in es:
                edge_left[e] = max(0.0, edge_left[e] - q)
            for cp in cps:
                key = (cp, self.pools[k])
                cp_left[key] = max(0.0, cp_left.get(key, 0.0) - q)
            add_projection(pair, week + int(np.ceil(lead)), q)

        return {'flows': flows}