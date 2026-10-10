# 0.4949796506031714
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
        self.pool = s['commodities'].get('pool', ['tb'] * len(self.values))
        self.stocks = [tuple(map(int, p)) for p in l['stock_slots']]
        self.supplies = [tuple(map(int, p)) for p in l['supply_slots']]
        self.demands = [tuple(map(int, p)) for p in l['demands']]
        self.di = {p: i for i, p in enumerate(self.demands)}
        self.cp_index = {int(n): i for i, n in enumerate(l['chokepoints'])}
        self.memory = {}

        sink_pi = {(int(n), int(k)): float(pi) for n, k, pi in
                   zip(s['sinks']['node'], s['sinks']['k'], s['sinks']['pi'])}
        self.pi = np.asarray([sink_pi.get(p, 1.0) for p in self.demands])
        self.maxpi = max(1.0, float(self.pi.max()) if self.pi.size else 1.0)

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
            alt = edges['alt_of'][edge] if lane < 0 else s['lanes']['alt_of'][lane]
            self.routes.append((i, k, es, cps, src, dst))
            grouped.setdefault((src, k, dst), []).append((alt, min(self.u0[e] for e in es)))

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

        self.dyad_regions = []
        for a, b in zip(s.get('dyads', {}).get('a', []), s.get('dyads', {}).get('b', [])):
            self.dyad_regions.append({int(a), int(b)})

    def field(self, o, name, fallback, remember=False):
        fb = np.asarray(fallback)
        old = self.memory.get(name, fb) if remember else fb
        if name not in o:
            x = np.array(old, copy=True)
        else:
            x = np.asarray(o[name])
            mask = o.get(name + '.observed')
            if mask is not None and np.shape(mask) == x.shape:
                x = np.where(mask, x, old)
            else:
                x = x.copy()
        if remember:
            self.memory[name] = np.array(x, copy=True)
        return x

    def act(self, o):
        week = int(np.asarray(o['week']).reshape(-1)[0])
        ne, nk = len(self.tail), len(self.values)
        ns = len(self.s['action_slots']['edge'])
        u = self.field(o, 'graph_now.u', self.u0, True).astype(float)
        c = self.field(o, 'graph_now.c', self.c0, True).astype(float)
        tau = np.maximum(1.0, self.field(o, 'graph_now.tau', self.tau0, True).astype(float))
        tariff = self.field(o, 'graph_now.tariff', np.zeros((ne, nk)), True).astype(float)
        banned = self.field(o, 'graph_now.prohibited', np.zeros((ne, nk)), True).astype(bool)
        opened = self.field(o, 'graph_now.open', np.ones(len(self.cp_index)), True).astype(float)
        caps = {p: self.field(o, 'graph_now.kappa.' + p, np.full(len(self.cp_index), np.inf), True).astype(float)
                for p in ('tb', 'ct')}
        mask = np.asarray(o.get('action_mask', np.ones(ns)), dtype=float).reshape(-1)

        available, stock, projected = {}, {}, {}
        stock_qty = self.field(o, 'stock.qty', np.zeros(len(self.stocks)))
        for p, q in zip(self.stocks, stock_qty):
            q = max(0.0, float(q))
            available[p] = stock[p] = q
            projected.setdefault(p, []).append((week, q))
        supply_qty = self.field(o, 'graph_now.supply.avail', np.zeros(len(self.supplies)))
        for p, q in zip(self.supplies, supply_qty):
            available[p] = available.get(p, 0.0) + max(0.0, float(q))

        lanes = self.s['lanes']['edges']
        pq = np.asarray(o.get('pipeline.qty', [])).reshape(-1)
        live = np.asarray(o.get('pipeline.qty.observed', np.ones(pq.size)), dtype=bool).reshape(-1)
        pe = np.asarray(o.get('pipeline.edge', np.zeros(pq.size)), dtype=int).reshape(-1)
        pk = np.asarray(o.get('pipeline.k', np.zeros(pq.size)), dtype=int).reshape(-1)
        pl = np.asarray(o.get('pipeline.lane', np.full(pq.size, -1)), dtype=int).reshape(-1)
        plm = np.asarray(o.get('pipeline.lane.observed', np.ones(pq.size)), dtype=bool).reshape(-1)
        pa = np.asarray(o.get('pipeline.arrival_week', np.full(pq.size, week)), dtype=int).reshape(-1)
        for j in range(min(pq.size, pe.size, pk.size, pa.size)):
            if j >= live.size or not live[j] or pq[j] <= 0 or not 0 <= pe[j] < ne:
                continue
            lane = int(pl[j]) if j < pl.size and j < plm.size and plm[j] else -1
            dst, arrival = int(self.head[pe[j]]), int(pa[j])
            if 0 <= lane < len(lanes):
                es = [int(x) for x in lanes[lane]]
                dst = int(self.head[es[-1]])
                if pe[j] in es:
                    arrival += int(sum(tau[e] for e in es[es.index(pe[j]) + 1:]))
            projected.setdefault((dst, int(pk[j])), []).append((max(week, arrival), float(pq[j])))

        wq = np.asarray(o.get('wip.qty', [])).reshape(-1)
        wm = np.asarray(o.get('wip.qty.observed', np.ones(wq.size)), dtype=bool).reshape(-1)
        wn = np.asarray(o.get('wip.node', np.zeros(wq.size)), dtype=int).reshape(-1)
        wk = np.asarray(o.get('wip.k', np.zeros(wq.size)), dtype=int).reshape(-1)
        wo = np.asarray(o.get('wip.out_week', np.full(wq.size, week)), dtype=int).reshape(-1)
        for j in range(min(wq.size, wn.size, wk.size, wo.size)):
            if j < wm.size and wm[j] and wq[j] > 0:
                projected.setdefault((int(wn[j]), int(wk[j])), []).append((max(week, int(wo[j])), float(wq[j])))

        lots = np.asarray(o.get('queue_lots.qty', np.zeros((0, 0))))
        lot_mask = np.asarray(o.get('queue_lots.qty.observed', np.ones(lots.shape)), dtype=bool)
        for row, key in enumerate(self.l.get('lot_keys', [])[:lots.shape[0]]):
            cp, k, lane, edge = key
            edge = int(edge)
            if not 0 <= edge < ne:
                continue
            es = [int(x) for x in lanes[int(lane)]] if lane is not None and 0 <= int(lane) < len(lanes) else [edge]
            dst = int(self.head[es[-1]])
            rem = sum(tau[e] for e in es[es.index(edge):]) if edge in es else tau[edge]
            for col in range(lots.shape[1]):
                if row < lot_mask.shape[0] and col < lot_mask.shape[1] and lot_mask[row, col] and lots[row, col] > 0:
                    projected.setdefault((dst, int(k)), []).append((max(week, col + 1) + int(np.ceil(rem)), float(lots[row, col])))

        pending = {}
        pe = np.asarray(o.get('pending_prohibitions.edge', []), dtype=int).reshape(-1)
        pkp = np.asarray(o.get('pending_prohibitions.k', np.zeros(pe.size)), dtype=int).reshape(-1)
        pw = np.asarray(o.get('pending_prohibitions.effective_week', np.full(pe.size, self.T + 1)), dtype=int).reshape(-1)
        pm = np.asarray(o.get('pending_prohibitions.edge.observed', np.ones(pe.size)), dtype=bool).reshape(-1)
        for j in range(min(pe.size, pkp.size, pw.size)):
            if j < pm.size and pm[j]:
                key = (int(pe[j]), int(pkp[j]))
                pending[key] = min(pending.get(key, self.T + 1), int(pw[j]))

        forecast = np.asarray(o.get('demand_forecast.qty', np.zeros((len(self.demands), 1))), dtype=float)
        backlog = np.asarray(o.get('backlog.qty', np.zeros(len(self.demands))), dtype=float).reshape(-1)
        warning = np.asarray(o.get('warning.score', np.zeros(len(self.l.get('warning_units', [])))), dtype=float).reshape(-1)
        warn_map = {tuple(key): float(np.clip(warning[i], 0.0, 1.0))
                    for i, key in enumerate(self.l.get('warning_units', [])) if i < warning.size}

        def covered(p, cutoff):
            return sum(q for arrival, q in projected.get(p, ()) if arrival <= cutoff)

        def need(r, lead):
            _, k, _, _, _, dst = r
            p = (dst, k)
            if p in self.di:
                di = self.di[p]
                h = min(forecast.shape[1] if forecast.ndim > 1 else 1,
                        max(1, int(np.ceil(lead)) + 2), max(1, self.T - week + 1))
                d = float(backlog[di]) if di < backlog.size else 0.0
                if forecast.ndim > 1 and di < forecast.shape[0]:
                    d += float(np.sum(forecast[di, :h]))
                return max(0.0, d - covered(p, week + h - 1))
            duration = min(max(2.0, lead + 2.0), max(1.0, self.T - week + 1))
            cutoff = week + int(np.ceil(duration))
            return max(0.0, self.node_rate.get(p, 0.0) * duration - covered(p, cutoff))

        edge_left = np.maximum(0.0, u.copy())
        cp_left = {}
        for cp, ci in self.cp_index.items():
            for pool in ('tb', 'ct'):
                cap = float(caps[pool][ci]) if ci < len(caps[pool]) else np.inf
                cp_left[(cp, pool)] = max(0.0, cap * max(0.0, opened[ci]))

        candidates = []
        node_regions = self.s['nodes'].get('region', [])
        for r in self.routes:
            i, k, es, cps, src, dst = r
            if i >= mask.size or mask[i] <= 0 or any(banned[e, k] for e in es):
                continue
            lead = float(sum(tau[e] for e in es))
            if week + lead > self.T:
                continue
            elapsed, risky = 0.0, False
            for e in es:
                if pending.get((e, k), self.T + 1) <= week + elapsed:
                    risky = True
                elapsed += tau[e]
            if risky:
                continue
            cap = min((max(0.0, float(u[e])) for e in es), default=0.0)
            for cp in cps:
                if cp in self.cp_index:
                    cap = min(cap, cp_left.get((cp, self.pool[k]), 0.0))
            if cap <= 0 or need(r, lead) <= 0:
                continue
            p = (dst, k)
            priority = max(1.0, self.priority.get(p, self.maxpi * 0.25))
            cost = sum(float(c[e]) + float(tariff[e, k]) * float(self.values[k]) for e in es)
            regions = {int(node_regions[n]) for n in (src, dst) if n < len(node_regions)}
            warnings = [warn_map.get(('chokepoint', cp), 0.0) for cp in cps]
            warnings += [warn_map.get(('region', region), 0.0) for region in regions]
            for di, pair in enumerate(self.dyad_regions):
                if pair.issubset(regions):
                    warnings.append(warn_map.get(('dyad', di), 0.0))
            risk = max(warnings, default=0.0)
            score = (cost + 0.002 * self.values[k] * lead) / priority + 0.08 * risk
            candidates.append((score, lead, i, cap, r))

        flows = np.zeros(ns, dtype=float)
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        for _, lead, i, planned_cap, r in candidates:
            _, k, es, cps, src, dst = r
            q = min(planned_cap, need(r, lead), max(0.0, available.get((src, k), 0.0)))
            for e in es:
                q = min(q, max(0.0, edge_left[e]))
            for cp in cps:
                if cp in self.cp_index:
                    q = min(q, cp_left.get((cp, self.pool[k]), 0.0))
            if q <= 0 or not np.isfinite(q):
                continue
            flows[i] = q
            available[(src, k)] = max(0.0, available.get((src, k), 0.0) - q)
            for e in es:
                edge_left[e] = max(0.0, edge_left[e] - q)
            for cp in cps:
                if cp in self.cp_index:
                    key = (cp, self.pool[k])
                    cp_left[key] = max(0.0, cp_left.get(key, 0.0) - q)
            projected.setdefault((dst, k), []).append((week + int(np.ceil(lead)), q))

        return {'flows': flows}
