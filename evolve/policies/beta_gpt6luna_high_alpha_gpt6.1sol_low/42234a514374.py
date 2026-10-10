# 0.5389396075457594
import numpy as np

class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.T = int(config['T'])
        e = self.s['edges']
        self.tail = np.asarray(e['tail'], dtype=int)
        self.head = np.asarray(e['head'], dtype=int)
        self.u0 = np.asarray([0.0 if x is None else x for x in e['u0']], dtype=float)
        self.c0 = np.asarray([0.0 if x is None else x for x in e['c0']], dtype=float)
        self.tau0 = np.asarray([1.0 if x is None else x for x in e['tau0']], dtype=float)
        self.values = np.asarray(self.s['commodities']['v'], dtype=float)
        self.pool = self.s['commodities'].get('pool', ['tb'] * len(self.values))
        self.stock_slots = [tuple(map(int, p)) for p in self.l['stock_slots']]
        self.supply_slots = [tuple(map(int, p)) for p in self.l['supply_slots']]
        self.demands = [tuple(map(int, p)) for p in self.l['demands']]
        self.demand_index = {p: i for i, p in enumerate(self.demands)}
        self.cp_index = {int(n): i for i, n in enumerate(self.l['chokepoints'])}
        self.routes = []
        grouped = {}
        slots = self.s['action_slots']
        for i, (edge, k, lane) in enumerate(zip(slots['edge'], slots['k'], slots['lane'])):
            edge, k = int(edge), int(k)
            lane_id = -1 if lane is None else int(lane)
            if lane_id < 0:
                es = [edge]
                cps = []
                alt = e['alt_of'][edge]
            else:
                es = [int(x) for x in self.s['lanes']['edges'][lane_id]]
                cps = [int(x) for x in self.s['lanes']['chokepoints'][lane_id]]
                alt = self.s['lanes']['alt_of'][lane_id]
            if not es:
                continue
            src = int(self.tail[es[0]])
            dst = int(self.head[es[-1]])
            r = {'slot': i, 'k': k, 'edges': es, 'cps': cps,
                 'src': src, 'dst': dst, 'alt': alt}
            self.routes.append(r)
            cap = min((self.u0[x] for x in es), default=0.0)
            grouped.setdefault((src, k, dst), []).append((alt, cap))

        incoming, outgoing = {}, {}
        for (src, k, dst), choices in grouped.items():
            base = [cap for alt, cap in choices if alt is None]
            cap = sum(base) if base else max((cap for _, cap in choices), default=0.0)
            incoming[(dst, k)] = incoming.get((dst, k), 0.0) + cap
            outgoing[(src, k)] = outgoing.get((src, k), 0.0) + cap
        self.node_rate = {}
        for p in set(incoming) | set(outgoing):
            a, b = incoming.get(p), outgoing.get(p)
            self.node_rate[p] = min(a, b) if a is not None and b is not None else max(a or 0.0, b or 0.0)

        sink_pi = {(int(n), int(k)): float(pi) for n, k, pi in
                   zip(self.s['sinks']['node'], self.s['sinks']['k'], self.s['sinks']['pi'])}
        self.max_pi = max([1.0] + list(sink_pi.values()))
        reverse = {}
        for r in self.routes:
            reverse.setdefault((r['dst'], r['k']), set()).add(r['src'])
        self.priority = {}
        for (sink, k), pi in sink_pi.items():
            queue = [(sink, 0)]
            seen = {sink}
            for node, dist in queue:
                p = (node, k)
                self.priority[p] = max(self.priority.get(p, 0.0), pi / (1.0 + 0.12 * dist))
                for prev in reverse.get(p, ()):
                    if prev not in seen:
                        seen.add(prev)
                        queue.append((prev, dist + 1))

        self.node_region = self.s['nodes'].get('region', [])
        self.dyads = self.s.get('dyads', {})

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
        ne = len(self.tail)
        nk = len(self.values)
        nslot = len(self.s['action_slots']['edge'])
        u = self.field(o, 'graph_now.u', self.u0).astype(float)
        c = self.field(o, 'graph_now.c', self.c0).astype(float)
        tau = np.maximum(1.0, self.field(o, 'graph_now.tau', self.tau0).astype(float))
        tariff = self.field(o, 'graph_now.tariff', np.zeros((ne, nk))).astype(float)
        prohibited = self.field(o, 'graph_now.prohibited', np.zeros((ne, nk))).astype(bool)
        opened = self.field(o, 'graph_now.open', np.ones(len(self.cp_index))).astype(float)
        mask = np.asarray(o.get('action_mask', np.ones(nslot)), dtype=float).reshape(-1)

        warnings = {}
        scores = self.field(o, 'warning.score', np.zeros(len(self.l['warning_units']))).reshape(-1)
        for key, score in zip(self.l['warning_units'], scores):
            try:
                warnings[tuple(key)] = float(np.clip(score, 0.0, 1.0))
            except Exception:
                pass

        def exposure(r):
            vals = [warnings.get(('chokepoint', cp), 0.0) for cp in r['cps']]
            regions = []
            for node in (r['src'], r['dst']):
                if 0 <= node < len(self.node_region):
                    regions.append(int(self.node_region[node]))
            vals.extend(warnings.get(('region', reg), 0.0) for reg in regions)
            dyad_a = self.dyads.get('a', [])
            dyad_b = self.dyads.get('b', [])
            for i, (a, b) in enumerate(zip(dyad_a, dyad_b)):
                if int(a) in regions and int(b) in regions:
                    vals.append(warnings.get(('dyad', i), 0.0))
            return max(vals, default=0.0)

        available, stock, projected = {}, {}, {}
        stockv = self.field(o, 'stock.qty', np.zeros(len(self.stock_slots))).reshape(-1)
        for i, p in enumerate(self.stock_slots):
            q = max(0.0, float(stockv[i])) if i < stockv.size else 0.0
            available[p] = stock[p] = q
            if q:
                projected.setdefault(p, []).append((week, q))
        supply = self.field(o, 'graph_now.supply.avail', np.zeros(len(self.supply_slots))).reshape(-1)
        for i, p in enumerate(self.supply_slots):
            if i < supply.size:
                available[p] = available.get(p, 0.0) + max(0.0, float(supply[i]))

        lanes = self.s['lanes']['edges']
        def lane_destination(edge, lane):
            if lane is not None and 0 <= int(lane) < len(lanes):
                es = [int(x) for x in lanes[int(lane)]]
                if es:
                    return int(self.head[es[-1]]), es
            return int(self.head[edge]), [int(edge)]

        # Existing shipments are credited when they arrive at the route's endpoint.
        pq = np.asarray(o.get('pipeline.qty', []), dtype=float).reshape(-1)
        live = np.asarray(o.get('pipeline.qty.observed', np.ones(pq.size)), dtype=bool).reshape(-1)
        for j in range(min(pq.size, live.size)):
            if not live[j] or pq[j] <= 0:
                continue
            edge = int(o['pipeline.edge'][j])
            if edge < 0 or edge >= ne:
                continue
            lm = np.asarray(o.get('pipeline.lane.observed', np.zeros(pq.size)), dtype=bool).reshape(-1)
            lane = int(o['pipeline.lane'][j]) if j < lm.size and lm[j] else -1
            dst, es = lane_destination(edge, lane)
            try:
                pos = es.index(edge)
            except ValueError:
                pos = 0
            arrival = int(o['pipeline.arrival_week'][j]) + int(np.ceil(sum(tau[x] for x in es[pos + 1:])))
            k = int(o['pipeline.k'][j])
            projected.setdefault((dst, k), []).append((max(week, arrival), float(pq[j])))

        wqty = np.asarray(o.get('wip.qty', []), dtype=float).reshape(-1)
        wlive = np.asarray(o.get('wip.qty.observed', np.ones(wqty.size)), dtype=bool).reshape(-1)
        for j in range(min(wqty.size, wlive.size)):
            if wlive[j] and wqty[j] > 0:
                p = (int(o['wip.node'][j]), int(o['wip.k'][j]))
                t = int(o['wip.out_week'][j])
                projected.setdefault(p, []).append((max(week, t), float(wqty[j])))

        closure_end = {}
        ce = np.asarray(o.get('closure_end.chokepoint', []), dtype=int).reshape(-1)
        cem = np.asarray(o.get('closure_end.chokepoint.observed', np.zeros(ce.size)), dtype=bool).reshape(-1)
        ew = np.asarray(o.get('closure_end.end_week', np.zeros(ce.size)), dtype=int).reshape(-1)
        ewm = np.asarray(o.get('closure_end.end_week.observed', np.zeros(ce.size)), dtype=bool).reshape(-1)
        for j in range(min(ce.size, cem.size, ew.size, ewm.size)):
            if cem[j] and ewm[j]:
                closure_end[int(ce[j])] = int(ew[j]) + 1

        lots = np.asarray(o.get('queue_lots.qty', np.zeros((0, 0))), dtype=float)
        lotmask = np.asarray(o.get('queue_lots.qty.observed', np.ones(lots.shape)), dtype=bool)
        for row, key in enumerate(self.l.get('lot_keys', [])[:lots.shape[0]]):
            if len(key) < 4 or row >= lotmask.shape[0]:
                continue
            cp, k, lane, edge = key
            edge = int(edge)
            if edge < 0 or edge >= ne:
                continue
            q = float(np.sum(np.where(lotmask[row], lots[row], 0.0)))
            if q <= 0:
                continue
            dst, es = lane_destination(edge, -1 if lane is None else int(lane))
            try:
                pos = es.index(edge)
            except ValueError:
                pos = 0
            release = week
            ci = self.cp_index.get(int(cp))
            if ci is not None and opened[ci] <= 0.001:
                release = max(release, closure_end.get(int(cp), week + 3))
            arrival = release + int(np.ceil(sum(tau[x] for x in es[pos:])))
            projected.setdefault((dst, int(k)), []).append((arrival, q))

        pending = {}
        pe = np.asarray(o.get('pending_prohibitions.edge', []), dtype=int).reshape(-1)
        pem = np.asarray(o.get('pending_prohibitions.edge.observed', np.zeros(pe.size)), dtype=bool).reshape(-1)
        pk = np.asarray(o.get('pending_prohibitions.k', np.zeros(pe.size)), dtype=int).reshape(-1)
        pw = np.asarray(o.get('pending_prohibitions.effective_week', np.full(pe.size, self.T + 1)), dtype=int).reshape(-1)
        for j in range(min(pe.size, pem.size, pk.size, pw.size)):
            if pem[j]:
                key = (int(pe[j]), int(pk[j]))
                pending[key] = min(pending.get(key, self.T + 1), int(pw[j]))

        forecast = np.asarray(o.get('demand_forecast.qty', np.zeros((len(self.demands), 1))), dtype=float)
        backlog = np.asarray(o.get('backlog.qty', np.zeros(len(self.demands))), dtype=float).reshape(-1)

        def coverage(p, cutoff):
            return sum(q for t, q in projected.get(p, ()) if t <= cutoff)

        def requirement(r, lead, risk):
            p = (r['dst'], r['k'])
            if p in self.demand_index:
                di = self.demand_index[p]
                h = max(1, int(np.ceil(lead)) + 1 + int(np.ceil(1.5 * risk)))
                h = min(h, max(0, self.T - week + 1))
                row = forecast[di] if forecast.ndim > 1 and di < forecast.shape[0] else np.zeros(0)
                n = min(h, len(row))
                total = float(backlog[di]) if di < backlog.size else 0.0
                if n:
                    total += float(np.sum(row[:n]))
                    if h > n:
                        total += (h - n) * float(np.mean(row[-min(3, n):]))
                return total, week + max(0, h - 1)
            duration = min(max(1.0, lead + 2.0 + 2.0 * risk), max(1.0, self.T - week + 1))
            return max(0.0, float(self.node_rate.get(p, 0.0))) * duration, week + int(np.ceil(duration))

        edge_left = np.maximum(0.0, u.copy())
        cp_left = {}
        for pool_name in ('tb', 'ct'):
            caps = self.field(o, 'graph_now.kappa.' + pool_name, np.full(len(self.cp_index), np.inf)).astype(float)
            for cp, ci in self.cp_index.items():
                cap = caps[ci] if ci < len(caps) else np.inf
                cp_left[(cp, pool_name)] = max(0.0, float(cap) * max(0.0, opened[ci]))

        candidates = []
        for r in self.routes:
            i, k = r['slot'], r['k']
            es, cps = r['edges'], r['cps']
            if i >= mask.size or mask[i] <= 0 or any(prohibited[e, k] for e in es):
                continue
            lead = float(sum(tau[e] for e in es))
            if week + lead > self.T + 1:
                continue
            elapsed = 0.0
            if any(pending.get((e, k), self.T + 1) <= week + sum(tau[x] for x in es[:j])
                   for j, e in enumerate(es)):
                continue
            cap = min((max(0.0, float(u[e])) for e in es), default=0.0)
            for cp in cps:
                ci = self.cp_index.get(cp)
                if ci is not None:
                    cap = min(cap, cp_left.get((cp, self.pool[k]), 0.0))
            if cap <= 0:
                continue
            risk = exposure(r)
            target, cutoff = requirement(r, lead, risk)
            need = max(0.0, target - coverage((r['dst'], k), cutoff))
            if need <= 0:
                continue
            freight = sum(float(c[e]) + float(tariff[e, k]) * float(self.values[k]) for e in es)
            # Warnings gently penalize exposed routes; they are not treated as certain closures.
            freight *= 1.0 + 0.35 * risk
            freight += 0.002 * float(self.values[k]) * lead
            priority = max(1.0, float(self.priority.get((r['dst'], k), self.max_pi * 0.25)))
            urgency = 0.35 + 0.65 * min(1.0, need / max(target, 1e-9))
            score = freight / (priority * urgency)
            candidates.append((score, lead, i, cap, risk))

        flows = np.zeros(nslot, dtype=float)
        for _, lead, i, planned_cap, risk in sorted(candidates, key=lambda x: (x[0], x[1], x[2])):
            r = next((x for x in self.routes if x['slot'] == i), None)
            if r is None:
                continue
            k, es, cps = r['k'], r['edges'], r['cps']
            srcp, dstp = (r['src'], k), (r['dst'], k)
            target, cutoff = requirement(r, lead, risk)
            need = max(0.0, target - coverage(dstp, cutoff))
            q = min(planned_cap, need, max(0.0, available.get(srcp, 0.0)))
            for e in es:
                q = min(q, max(0.0, edge_left[e]))
            for cp in cps:
                q = min(q, cp_left.get((cp, self.pool[k]), np.inf))
            if q <= 0 or not np.isfinite(q):
                continue
            flows[i] = q
            available[srcp] = max(0.0, available.get(srcp, 0.0) - q)
            stock[srcp] = max(0.0, stock.get(srcp, 0.0) - q)
            for e in es:
                edge_left[e] = max(0.0, edge_left[e] - q)
            for cp in cps:
                key = (cp, self.pool[k])
                if key in cp_left:
                    cp_left[key] = max(0.0, cp_left[key] - q)
            projected.setdefault(dstp, []).append((week + int(np.ceil(lead)), q))

        return {'flows': flows}