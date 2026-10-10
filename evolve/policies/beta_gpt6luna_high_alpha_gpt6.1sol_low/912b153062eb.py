# -2.107804432305058
import heapq
import numpy as np


class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.T = int(config['T'])
        s, l = self.s, self.l
        edges = s['edges']
        self.tail = np.asarray(edges['tail'], dtype=int)
        self.head = np.asarray(edges['head'], dtype=int)
        self.u0 = np.asarray([0.0 if x is None else float(x) for x in edges['u0']])
        self.c0 = np.asarray([0.0 if x is None else float(x) for x in edges['c0']])
        self.tau0 = np.asarray([1.0 if x is None else float(x) for x in edges['tau0']])
        self.values = np.asarray(s['commodities']['v'], dtype=float)
        self.pools = s['commodities'].get('pool', ['tb'] * len(self.values))
        self.regions = s['nodes'].get('region', [])
        self.demands = [tuple(map(int, x)) for x in l['demands']]
        self.cp_index = {int(n): i for i, n in enumerate(l['chokepoints'])}
        self.demand_index = {p: i for i, p in enumerate(self.demands)}
        self.pi = np.asarray(s['sinks'].get('pi', np.ones(len(self.demands))), dtype=float)
        self.routes = []
        self.outgoing = {}
        self.incoming = {}
        slots = s['action_slots']
        for i, (edge, k, lane) in enumerate(zip(slots['edge'], slots['k'], slots['lane'])):
            edge, k = int(edge), int(k)
            lane = -1 if lane is None else int(lane)
            es = [edge] if lane < 0 else [int(x) for x in s['lanes']['edges'][lane]]
            cps = [] if lane < 0 else [int(x) for x in s['lanes']['chokepoints'][lane]]
            if not es:
                continue
            alt = edges['alt_of'][edge] if lane < 0 else s['lanes']['alt_of'][lane]
            r = {'slot': i, 'k': k, 'edges': es, 'cps': cps,
                 'src': int(self.tail[es[0]]), 'dst': int(self.head[es[-1]]),
                 'alt': alt, 'lane': lane}
            self.routes.append(r)
            self.outgoing.setdefault((r['src'], k), []).append(r)
            self.incoming.setdefault((r['dst'], k), []).append(r)

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

    @staticmethod
    def live(o, name):
        x = np.asarray(o.get(name, [])).reshape(-1)
        m = np.asarray(o.get(name + '.observed', np.ones(x.size)), dtype=bool).reshape(-1)
        return x, m

    def act(self, o):
        week = int(np.asarray(o['week']).reshape(-1)[0])
        ne, nk = len(self.tail), len(self.values)
        nslot = len(self.s['action_slots']['edge'])
        u = self.field(o, 'graph_now.u', self.u0).astype(float)
        c = self.field(o, 'graph_now.c', self.c0).astype(float)
        tau = np.maximum(1.0, self.field(o, 'graph_now.tau', self.tau0).astype(float))
        tariff = self.field(o, 'graph_now.tariff', np.zeros((ne, nk))).astype(float)
        prohibited = self.field(o, 'graph_now.prohibited', np.zeros((ne, nk))).astype(bool)
        opened = self.field(o, 'graph_now.open', np.ones(len(self.cp_index))).astype(float)
        mask = np.asarray(o.get('action_mask', np.ones(nslot)), dtype=bool).reshape(-1)

        warnings = {}
        wv = self.field(o, 'warning.score', np.zeros(len(self.l.get('warning_units', [])))).reshape(-1)
        for key, value in zip(self.l.get('warning_units', []), wv):
            try:
                warnings[tuple(key)] = float(np.clip(value, 0.0, 1.0))
            except Exception:
                pass
        dyads = self.s.get('dyads', {})
        dyad_a, dyad_b = dyads.get('a', []), dyads.get('b', [])

        stock_slots = [tuple(map(int, x)) for x in self.l['stock_slots']]
        supply_slots = [tuple(map(int, x)) for x in self.l['supply_slots']]
        available = {}
        stock = self.field(o, 'stock.qty', np.zeros(len(stock_slots))).reshape(-1)
        for i, p in enumerate(stock_slots):
            q = max(0.0, float(stock[i])) if i < stock.size else 0.0
            available[p] = available.get(p, 0.0) + q
        supply = self.field(o, 'graph_now.supply.avail', np.zeros(len(supply_slots))).reshape(-1)
        for i, p in enumerate(supply_slots):
            if i < supply.size:
                available[p] = available.get(p, 0.0) + max(0.0, float(supply[i]))

        projected = {}
        def add_projected(p, arrival, qty):
            if qty > 0 and np.isfinite(qty):
                projected.setdefault(p, []).append((int(arrival), float(qty)))

        lanes = self.s.get('lanes', {}).get('edges', [])
        pq, plive = self.live(o, 'pipeline.qty')
        pe = np.asarray(o.get('pipeline.edge', np.zeros(pq.size)), dtype=int).reshape(-1)
        pk = np.asarray(o.get('pipeline.k', np.zeros(pq.size)), dtype=int).reshape(-1)
        pl = np.asarray(o.get('pipeline.lane', np.full(pq.size, -1)), dtype=int).reshape(-1)
        plm = np.asarray(o.get('pipeline.lane.observed', np.ones(pq.size)), dtype=bool).reshape(-1)
        pa = np.asarray(o.get('pipeline.arrival_week', np.full(pq.size, week)), dtype=int).reshape(-1)
        for j in range(min(pq.size, pe.size, pk.size, pa.size)):
            if j >= plive.size or not plive[j] or pq[j] <= 0 or not 0 <= pe[j] < ne:
                continue
            lane = int(pl[j]) if j < pl.size and j < plm.size and plm[j] else -1
            dst, arrival = int(self.head[pe[j]]), int(pa[j])
            if 0 <= lane < len(lanes) and lanes[lane]:
                les = [int(x) for x in lanes[lane]]
                dst = int(self.head[les[-1]])
                if pe[j] in les:
                    pos = les.index(int(pe[j]))
                    arrival += int(np.ceil(sum(tau[e] for e in les[pos + 1:])))
            add_projected((dst, int(pk[j])), max(week, arrival), pq[j])

        wq, wlive = self.live(o, 'wip.qty')
        wn = np.asarray(o.get('wip.node', np.zeros(wq.size)), dtype=int).reshape(-1)
        wk = np.asarray(o.get('wip.k', np.zeros(wq.size)), dtype=int).reshape(-1)
        wo = np.asarray(o.get('wip.out_week', np.full(wq.size, week)), dtype=int).reshape(-1)
        for j in range(min(wq.size, wn.size, wk.size, wo.size)):
            if j < wlive.size and wlive[j] and wq[j] > 0:
                add_projected((int(wn[j]), int(wk[j])), max(week, int(wo[j])), wq[j])

        if 'queue_lots.qty' in o:
            lots = np.asarray(o['queue_lots.qty'], dtype=float)
            lm = np.asarray(o.get('queue_lots.qty.observed', np.ones(lots.shape)), dtype=bool)
            for row, key in enumerate(self.l.get('lot_keys', [])[:lots.shape[0]]):
                if len(key) < 4 or row >= lm.shape[0]:
                    continue
                cp, k, lane, next_edge = key
                next_edge = int(next_edge)
                if not 0 <= next_edge < ne:
                    continue
                lane = -1 if lane is None else int(lane)
                dst = int(self.head[next_edge])
                rem = float(tau[next_edge])
                if 0 <= lane < len(lanes) and lanes[lane]:
                    les = [int(x) for x in lanes[lane]]
                    dst = int(self.head[les[-1]])
                    if next_edge in les:
                        rem = float(sum(tau[e] for e in les[les.index(next_edge):]))
                for col in range(min(lots.shape[1], lm.shape[1])):
                    q = max(0.0, float(lots[row, col])) if lm[row, col] else 0.0
                    if q:
                        add_projected((dst, int(k)), max(week, col + 1) + int(np.ceil(rem)), q)

        pending = {}
        p_edge, p_live = self.live(o, 'pending_prohibitions.edge')
        p_k = np.asarray(o.get('pending_prohibitions.k', np.zeros(p_edge.size)), dtype=int).reshape(-1)
        p_week = np.asarray(o.get('pending_prohibitions.effective_week', np.full(p_edge.size, self.T + 1)), dtype=int).reshape(-1)
        for j in range(min(p_edge.size, p_k.size, p_week.size, p_live.size)):
            if p_live[j]:
                key = (int(p_edge[j]), int(p_k[j]))
                pending[key] = min(pending.get(key, self.T + 1), int(p_week[j]))

        forecast = np.asarray(o.get('demand_forecast.qty', np.zeros((len(self.demands), 1))), dtype=float)
        backlog = np.asarray(o.get('backlog.qty', np.zeros(len(self.demands))), dtype=float).reshape(-1)
        def coverage(p, cutoff):
            onhand = available.get(p, 0.0) if cutoff >= week else 0.0
            return onhand + sum(q for arrival, q in projected.get(p, []) if arrival <= cutoff)
        def sink_need(di, lead):
            horizon = min(max(1, self.T - week + 1), max(1, int(np.ceil(lead)) + 2))
            target = max(0.0, float(backlog[di])) if di < backlog.size else 0.0
            if forecast.ndim == 2 and di < forecast.shape[0] and forecast.shape[1]:
                row = forecast[di]
                used = min(horizon, row.size)
                target += float(np.sum(row[:used]))
                if horizon > used:
                    target += (horizon - used) * max(0.0, float(np.mean(row[-min(3, row.size):])))
            cutoff = week + horizon - 1
            p = self.demands[di]
            return max(0.0, target - coverage(p, cutoff)), cutoff

        cp_caps = {}
        for pool in ('tb', 'ct'):
            vals = self.field(o, 'graph_now.kappa.' + pool, np.full(len(self.cp_index), np.inf)).astype(float)
            for cp, ci in self.cp_index.items():
                cap = float(vals[ci]) if ci < vals.size else np.inf
                cp_caps[(cp, pool)] = max(0.0, cap * max(0.0, float(opened[ci])))
        edge_left = np.maximum(0.0, u.copy())
        usable = []
        outgoing = {}
        incoming = {}
        for old in self.routes:
            r = dict(old)
            i, k = r['slot'], r['k']
            if i >= mask.size or not mask[i] or not r['edges']:
                continue
            if any(e < 0 or e >= ne or (k < prohibited.shape[1] and prohibited[e, k]) for e in r['edges']):
                continue
            lead = float(sum(tau[e] for e in r['edges']))
            if week + lead > self.T + 1:
                continue
            cap = min((max(0.0, float(u[e])) for e in r['edges']), default=0.0)
            for cp in r['cps']:
                ci = self.cp_index.get(cp)
                if ci is not None:
                    cap = min(cap, cp_caps.get((cp, self.pools[k]), 0.0))
            if cap <= 0:
                continue
            freight = 0.0
            for e in r['edges']:
                freight += float(c[e]) + (float(tariff[e, k]) * float(self.values[k]) if e < tariff.shape[0] and k < tariff.shape[1] else 0.0)
            risk = max([warnings.get(('chokepoint', cp), 0.0) for cp in r['cps']] + [0.0])
            regs = []
            for node in (r['src'], r['dst']):
                if 0 <= node < len(self.regions):
                    regs.append(int(self.regions[node]))
            risk = max([risk] + [warnings.get(('region', reg), 0.0) for reg in regs])
            for j, (a, b) in enumerate(zip(dyad_a, dyad_b)):
                if int(a) in regs and int(b) in regs:
                    risk = max(risk, warnings.get(('dyad', j), 0.0))
            cost = freight + 0.12 * risk * max(1.0, freight) + 0.001 * risk * float(self.values[k])
            cost += 0.002 * float(self.values[k]) * lead
            r.update({'lead': lead, 'cap': cap, 'cost': max(0.0, cost), 'risk': risk})
            usable.append(r)
            outgoing.setdefault((r['src'], k), []).append(r)
            incoming.setdefault((r['dst'], k), []).append(r)

        candidates = []
        for (src, k), qty in list(available.items()):
            if qty <= 1e-9:
                continue
            sinks = [(di, sink) for di, (sink, dk) in enumerate(self.demands) if dk == k and sink != src]
            for di, sink in sinks:
                # Reverse Dijkstra gives the cheapest continuation from each node;
                # excluding src prevents a forced first route from making a cycle.
                dist = {sink: 0.0}
                nxt = {}
                heap = [(0.0, sink)]
                while heap:
                    d0, node = heapq.heappop(heap)
                    if d0 > dist.get(node, float('inf')) + 1e-10:
                        continue
                    for r in incoming.get((node, k), []):
                        if r['src'] == src:
                            continue
                        nd = d0 + r['cost']
                        if nd + 1e-10 < dist.get(r['src'], float('inf')):
                            dist[r['src']] = nd
                            nxt[r['src']] = r
                            heapq.heappush(heap, (nd, r['src']))
                for first in outgoing.get((src, k), []):
                    if first['dst'] != sink and first['dst'] not in dist:
                        continue
                    path = [first]
                    node = first['dst']
                    seen = {src}
                    valid = True
                    while node != sink:
                        if node in seen or node not in nxt:
                            valid = False
                            break
                        seen.add(node)
                        r = nxt[node]
                        path.append(r)
                        node = r['dst']
                    if not valid:
                        continue
                    lead = sum(r['lead'] for r in path)
                    if week + lead > self.T + 1:
                        continue
                    elapsed = 0.0
                    for r in path:
                        for e in r['edges']:
                            if pending.get((e, k), self.T + 1) <= week + elapsed:
                                valid = False
                                break
                            elapsed += tau[e]
                        if not valid:
                            break
                    if not valid:
                        continue
                    need, _ = sink_need(di, lead)
                    if need <= 1e-9:
                        continue
                    pi = max(1.0, float(self.pi[di]) if di < self.pi.size else 1.0)
                    score = sum(r['cost'] for r in path) / pi + 0.006 * lead
                    candidates.append((score, lead, first, di, path))

        flows = np.zeros(nslot, dtype=float)
        for _, lead, first, di, path in sorted(candidates, key=lambda x: (x[0], x[1], x[2]['slot'])):
            k = first['k']
            need, _ = sink_need(di, lead)
            q = min(first['cap'], need, max(0.0, available.get((first['src'], k), 0.0)))
            for e in first['edges']:
                q = min(q, max(0.0, edge_left[e]))
            for cp in first['cps']:
                if cp in self.cp_index:
                    q = min(q, cp_caps.get((cp, self.pools[k]), 0.0))
            if q <= 1e-9 or not np.isfinite(q):
                continue
            flows[first['slot']] += q
            available[(first['src'], k)] = max(0.0, available.get((first['src'], k), 0.0) - q)
            for e in first['edges']:
                edge_left[e] = max(0.0, edge_left[e] - q)
            for cp in first['cps']:
                key = (cp, self.pools[k])
                if key in cp_caps:
                    cp_caps[key] = max(0.0, cp_caps[key] - q)
            add_projected((first['dst'], k), week + int(np.ceil(first['lead'])), q)
            if first['dst'] != self.demands[di][0]:
                add_projected(self.demands[di], week + int(np.ceil(lead)), q)

        return {'flows': flows}