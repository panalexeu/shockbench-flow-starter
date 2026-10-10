# -1.940920842588046
import heapq
import numpy as np


class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.T = int(config['T'])
        s, l = self.s, self.l
        e = s['edges']
        self.tail = np.asarray(e['tail'], dtype=int)
        self.head = np.asarray(e['head'], dtype=int)
        self.u0 = np.asarray([0.0 if x is None else float(x) for x in e['u0']])
        self.c0 = np.asarray([0.0 if x is None else float(x) for x in e['c0']])
        self.tau0 = np.asarray([1.0 if x is None else float(x) for x in e['tau0']])
        self.values = np.asarray(s['commodities']['v'], dtype=float)
        self.pool = s['commodities'].get('pool', ['tb'] * len(self.values))
        self.stock_slots = [tuple(map(int, x)) for x in l.get('stock_slots', [])]
        self.supply_slots = [tuple(map(int, x)) for x in l.get('supply_slots', [])]
        self.demands = [tuple(map(int, x)) for x in l.get('demands', [])]
        self.demand_index = {p: i for i, p in enumerate(self.demands)}
        self.cp_index = {int(n): i for i, n in enumerate(l.get('chokepoints', []))}
        self.pi = np.asarray(s.get('sinks', {}).get('pi', np.ones(len(self.demands))), dtype=float)
        self.max_pi = max(1.0, float(np.max(self.pi)) if self.pi.size else 1.0)
        self.node_type = s.get('nodes', {}).get('type', [])
        self.node_region = s.get('nodes', {}).get('region', [])
        self.dyads = s.get('dyads', {})

        # A nominal throughput estimate is used only for small fab/OSAT buffers.
        self.routes = []
        grouped = {}
        slots = s.get('action_slots', {})
        for i, (edge, k, lane) in enumerate(zip(slots.get('edge', []), slots.get('k', []), slots.get('lane', []))):
            edge, k = int(edge), int(k)
            lane_id = -1 if lane is None else int(lane)
            if lane_id < 0:
                es = [edge]
                cps = []
                alt = e.get('alt_of', [None] * len(self.tail))[edge]
            else:
                es = [int(x) for x in s.get('lanes', {}).get('edges', [])[lane_id]]
                cps = [int(x) for x in s.get('lanes', {}).get('chokepoints', [])[lane_id]]
                alt = s.get('lanes', {}).get('alt_of', [None] * (lane_id + 1))[lane_id]
            if not es:
                continue
            src, dst = int(self.tail[es[0]]), int(self.head[es[-1]])
            cap = min((self.u0[x] for x in es), default=0.0)
            self.routes.append({'i': i, 'k': k, 'edges': es, 'cps': cps,
                                'src': src, 'dst': dst, 'alt': alt})
            grouped.setdefault((src, k, dst), []).append((alt, cap))

        incoming, outgoing = {}, {}
        for (src, k, dst), choices in grouped.items():
            base = [cap for alt, cap in choices if alt is None]
            cap = sum(base) if base else max((x[1] for x in choices), default=0.0)
            incoming[(dst, k)] = incoming.get((dst, k), 0.0) + cap
            outgoing[(src, k)] = outgoing.get((src, k), 0.0) + cap
        self.node_rate = {}
        for p in set(incoming) | set(outgoing):
            a, b = incoming.get(p), outgoing.get(p)
            self.node_rate[p] = min(a, b) if a is not None and b is not None else max(a or 0.0, b or 0.0)

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
        nslot = len(self.s.get('action_slots', {}).get('edge', []))
        u = self.field(o, 'graph_now.u', self.u0).astype(float).reshape(-1)
        c = self.field(o, 'graph_now.c', self.c0).astype(float).reshape(-1)
        tau = np.maximum(1.0, self.field(o, 'graph_now.tau', self.tau0).astype(float).reshape(-1))
        tariff = self.field(o, 'graph_now.tariff', np.zeros((ne, nk))).astype(float)
        prohibited = self.field(o, 'graph_now.prohibited', np.zeros((ne, nk))).astype(bool)
        mask = np.asarray(o.get('action_mask', np.ones(nslot)), dtype=bool).reshape(-1)
        opened = self.field(o, 'graph_now.open', np.ones(len(self.cp_index))).astype(float).reshape(-1)
        warnings = {}
        ws = self.field(o, 'warning.score', np.zeros(len(self.l.get('warning_units', [])))).reshape(-1)
        for item, val in zip(self.l.get('warning_units', []), ws):
            try:
                warnings[(str(item[0]), int(item[1]))] = float(np.clip(val, 0.0, 1.0))
            except Exception:
                pass

        def exposure(route):
            vals = [warnings.get(('chokepoint', int(cp)), 0.0) for cp in route['cps']]
            regs = []
            for node in (route['src'], route['dst']):
                if 0 <= node < len(self.node_region):
                    regs.append(int(self.node_region[node]))
            vals.extend(warnings.get(('region', r), 0.0) for r in regs)
            for j, (a, b) in enumerate(zip(self.dyads.get('a', []), self.dyads.get('b', []))):
                if int(a) in regs and int(b) in regs:
                    vals.append(warnings.get(('dyad', j), 0.0))
            return max(vals, default=0.0)

        stock = self.field(o, 'stock.qty', np.zeros(len(self.stock_slots))).astype(float).reshape(-1)
        available = {}
        projected = {}
        for j, p in enumerate(self.stock_slots):
            q = max(0.0, float(stock[j])) if j < stock.size else 0.0
            available[p] = q
            if q > 0:
                projected.setdefault(p, []).append((week, q))
        supply = self.field(o, 'graph_now.supply.avail', np.zeros(len(self.supply_slots))).astype(float).reshape(-1)
        for j, p in enumerate(self.supply_slots):
            if j < supply.size:
                available[p] = available.get(p, 0.0) + max(0.0, float(supply[j]))

        # Existing pipeline arrivals, including the remaining edges of a lane.
        pq = np.asarray(o.get('pipeline.qty', [])).reshape(-1)
        plive = np.asarray(o.get('pipeline.qty.observed', np.ones(pq.size)), dtype=bool).reshape(-1)
        pe = np.asarray(o.get('pipeline.edge', np.zeros(pq.size)), dtype=int).reshape(-1)
        pk = np.asarray(o.get('pipeline.k', np.zeros(pq.size)), dtype=int).reshape(-1)
        pl = np.asarray(o.get('pipeline.lane', np.full(pq.size, -1)), dtype=int).reshape(-1)
        plo = np.asarray(o.get('pipeline.lane.observed', np.ones(pq.size)), dtype=bool).reshape(-1)
        pa = np.asarray(o.get('pipeline.arrival_week', np.full(pq.size, week)), dtype=int).reshape(-1)
        lane_table = self.s.get('lanes', {}).get('edges', [])
        for j in range(min(pq.size, pe.size, pk.size, pa.size)):
            if j >= plive.size or not plive[j] or pq[j] <= 0 or not 0 <= pe[j] < ne:
                continue
            lane = int(pl[j]) if j < pl.size and j < plo.size and plo[j] else -1
            dst, arrival = int(self.head[pe[j]]), int(pa[j])
            if 0 <= lane < len(lane_table):
                les = [int(x) for x in lane_table[lane]]
                if les:
                    dst = int(self.head[les[-1]])
                    if int(pe[j]) in les:
                        pos = les.index(int(pe[j]))
                        arrival += int(np.ceil(sum(tau[x] for x in les[pos + 1:])))
            projected.setdefault((dst, int(pk[j])), []).append((max(week, arrival), float(pq[j])))

        # Work in process is available at its stated output node and week.
        wq = np.asarray(o.get('wip.qty', [])).reshape(-1)
        wl = np.asarray(o.get('wip.qty.observed', np.ones(wq.size)), dtype=bool).reshape(-1)
        wn = np.asarray(o.get('wip.node', np.zeros(wq.size)), dtype=int).reshape(-1)
        wk = np.asarray(o.get('wip.k', np.zeros(wq.size)), dtype=int).reshape(-1)
        wo = np.asarray(o.get('wip.out_week', np.full(wq.size, week)), dtype=int).reshape(-1)
        for j in range(min(wq.size, wn.size, wk.size, wo.size)):
            if j < wl.size and wl[j] and wq[j] > 0:
                projected.setdefault((int(wn[j]), int(wk[j])), []).append((max(week, int(wo[j])), float(wq[j])))

        # Chokepoint queue lots are inbound inventory too.
        lots = np.asarray(o.get('queue_lots.qty', np.zeros((0, 0))), dtype=float)
        lm = np.asarray(o.get('queue_lots.qty.observed', np.ones(lots.shape)), dtype=bool)
        for row, key in enumerate(self.l.get('lot_keys', [])[:lots.shape[0]]):
            if len(key) < 4:
                continue
            _, k, lane, edge = key
            edge = int(edge)
            if not 0 <= edge < ne:
                continue
            lane = -1 if lane is None else int(lane)
            if 0 <= lane < len(lane_table) and lane_table[lane]:
                les = [int(x) for x in lane_table[lane]]
                dst = int(self.head[les[-1]])
                rem = sum(tau[x] for x in les[les.index(edge):]) if edge in les else tau[edge]
            else:
                dst, rem = int(self.head[edge]), tau[edge]
            for col in range(lots.shape[1]):
                if row < lm.shape[0] and col < lm.shape[1] and lm[row, col] and lots[row, col] > 0:
                    projected.setdefault((dst, int(k)), []).append((max(week, col + 1) + int(np.ceil(rem)), float(lots[row, col])))

        forecast = np.asarray(o.get('demand_forecast.qty', np.zeros((len(self.demands), 1))), dtype=float)
        backlog = np.asarray(o.get('backlog.qty', np.zeros(len(self.demands))), dtype=float).reshape(-1)
        pending = {}
        p_edge = np.asarray(o.get('pending_prohibitions.edge', []), dtype=int).reshape(-1)
        p_k = np.asarray(o.get('pending_prohibitions.k', np.zeros(p_edge.size)), dtype=int).reshape(-1)
        p_week = np.asarray(o.get('pending_prohibitions.effective_week', np.full(p_edge.size, self.T + 1)), dtype=int).reshape(-1)
        p_live = np.asarray(o.get('pending_prohibitions.edge.observed', np.ones(p_edge.size)), dtype=bool).reshape(-1)
        for j in range(min(p_edge.size, p_k.size, p_week.size)):
            if j < p_live.size and p_live[j]:
                key = (int(p_edge[j]), int(p_k[j]))
                pending[key] = min(pending.get(key, self.T + 1), int(p_week[j]))

        ktb = self.field(o, 'graph_now.kappa.tb', np.full(len(self.cp_index), np.inf)).astype(float).reshape(-1)
        kct = self.field(o, 'graph_now.kappa.ct', np.full(len(self.cp_index), np.inf)).astype(float).reshape(-1)
        cp_left = {}
        for cp, ci in self.cp_index.items():
            op = max(0.0, float(opened[ci])) if ci < opened.size else 1.0
            for name, arr in (('tb', ktb), ('ct', kct)):
                cap = float(arr[ci]) if ci < arr.size else np.inf
                cp_left[(cp, name)] = max(0.0, cap * op)
        edge_left = np.maximum(0.0, u.copy())

        # Build route arcs that are legal this week and not known to close
        # before the shipment reaches the relevant edge.
        arc_list = []
        for r in self.routes:
            i, k, es, cps = r['i'], r['k'], r['edges'], r['cps']
            if i >= mask.size or not mask[i] or not es or k < 0 or k >= nk:
                continue
            if any(e < 0 or e >= ne or e >= prohibited.shape[0] or k >= prohibited.shape[1] or prohibited[e, k] for e in es):
                continue
            lead = float(sum(tau[e] for e in es))
            if week + lead > self.T + 1:
                continue
            elapsed = 0.0
            invalid = False
            for e in es:
                if pending.get((e, k), self.T + 1) <= week + elapsed:
                    invalid = True
                    break
                elapsed += tau[e]
            if invalid:
                continue
            cap = min((max(0.0, float(u[e])) for e in es), default=0.0)
            for cp in cps:
                ci = self.cp_index.get(cp)
                if ci is not None:
                    cap = min(cap, cp_left.get((cp, self.pool[k]), 0.0))
            if cap <= 0:
                continue
            cost = 0.0
            for e in es:
                tc = float(tariff[e, k]) if e < tariff.shape[0] and k < tariff.shape[1] else 0.0
                cost += float(c[e]) + tc * float(self.values[k])
            risk = exposure(r)
            cost = max(0.0, cost) * (1.0 + 0.25 * risk)
            cost += 0.002 * float(self.values[k]) * lead
            r.update({'lead': lead, 'cost': cost, 'cap': cap})
            arc_list.append(r)

        adj = {}
        for r in arc_list:
            adj.setdefault((r['src'], r['k']), []).append(r)

        def forecast_total(di, cutoff):
            total = max(0.0, float(backlog[di])) if di < backlog.size else 0.0
            h = max(0, int(cutoff) - week + 1)
            if forecast.ndim != 2 or di >= forecast.shape[0] or forecast.shape[1] == 0:
                return total
            row = forecast[di]
            n = min(h, row.size)
            total += float(np.sum(row[:n]))
            if h > n:
                tail = row[-min(3, row.size):]
                total += (h - n) * max(0.0, float(np.mean(tail)))
            return total

        def projected_at(p, cutoff):
            return sum(q for arrival, q in projected.get(p, []) if arrival <= cutoff)

        # Include explicitly demanded sinks plus modest input buffers at fabs/OSATs.
        targets = []
        for di, p in enumerate(self.demands):
            targets.append({'p': p, 'kind': 'sink', 'di': di})
        for p, rate in self.node_rate.items():
            node, k = p
            if p in self.demand_index or rate <= 0 or node < 0 or node >= len(self.node_type):
                continue
            if self.node_type[node] in ('fab', 'osat'):
                targets.append({'p': p, 'kind': 'buffer', 'di': -1})

        flows = np.zeros(nslot, dtype=float)
        used_first = set()
        # Each slot is assigned at most once to avoid repeatedly planning the same
        # end-to-end path against downstream capacity that is not reserved today.
        for _ in range(nslot):
            candidates = []
            for source, qty in list(available.items()):
                src, k = source
                if qty <= 1e-9:
                    continue
                arcs = adj.get((src, k), [])
                if not arcs:
                    continue
                dist = {src: 0.0}
                prev = {}
                heap = [(0.0, src)]
                while heap:
                    d, node = heapq.heappop(heap)
                    if d > dist.get(node, float('inf')) + 1e-10:
                        continue
                    for r in adj.get((node, k), []):
                        nd = d + r['cost'] + 1e-9
                        if nd < dist.get(r['dst'], float('inf')):
                            dist[r['dst']] = nd
                            prev[r['dst']] = (node, r)
                            heapq.heappush(heap, (nd, r['dst']))
                for target in targets:
                    dest, tk = target['p']
                    if tk != k or dest not in dist or dest == src:
                        continue
                    path = []
                    node = dest
                    seen = set()
                    while node != src and node in prev and node not in seen:
                        seen.add(node)
                        parent, r = prev[node]
                        path.append(r)
                        node = parent
                    if node != src or not path:
                        continue
                    path.reverse()
                    first = path[0]
                    if first['i'] in used_first:
                        continue
                    lead = sum(r['lead'] for r in path)
                    arrival = week + int(np.ceil(lead))
                    if arrival > self.T:
                        continue
                    bottleneck = min(r['cap'] for r in path)
                    if target['kind'] == 'sink':
                        di = target['di']
                        horizon = min(max(1, int(np.ceil(lead)) + 2), max(1, self.T - week + 1))
                        cutoff = week + horizon - 1
                        need = max(0.0, forecast_total(di, cutoff) - projected_at(target['p'], cutoff))
                        priority = max(1.0, float(self.pi[di]) if di < self.pi.size else 1.0)
                    else:
                        rate = max(0.0, float(self.node_rate.get(target['p'], 0.0)))
                        cutoff = min(self.T, arrival + 1)
                        desired = rate * 1.5
                        need = max(0.0, desired - projected_at(target['p'], cutoff))
                        priority = max(1.0, self.max_pi * 0.04)
                    if need <= 1e-9:
                        continue
                    # The sink penalty rewards valuable demand; lead time also
                    # contributes through the path cost and the demand horizon.
                    score = dist[dest] / priority
                    candidates.append((score, lead, first['i'], source, target, path, bottleneck, need))

            if not candidates:
                break
            candidates.sort(key=lambda x: (x[0], x[1], x[2]))
            dispatched = False
            for _, lead, _, source, target, path, bottleneck, need in candidates:
                first = path[0]
                q = min(float(available.get(source, 0.0)), float(need), float(bottleneck))
                for e in first['edges']:
                    q = min(q, max(0.0, float(edge_left[e])))
                for cp in first['cps']:
                    q = min(q, cp_left.get((cp, self.pool[first['k']]), np.inf))
                if q <= 1e-9 or not np.isfinite(q):
                    continue
                flows[first['i']] += q
                available[source] = max(0.0, available.get(source, 0.0) - q)
                for e in first['edges']:
                    edge_left[e] = max(0.0, edge_left[e] - q)
                for cp in first['cps']:
                    key = (cp, self.pool[first['k']])
                    if key in cp_left:
                        cp_left[key] = max(0.0, cp_left[key] - q)
                used_first.add(first['i'])
                arrival = week + int(np.ceil(lead))
                projected.setdefault(target['p'], []).append((arrival, q))
                dispatched = True
                break
            if not dispatched:
                break

        return {'flows': flows}
