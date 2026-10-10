# 0.45436076670945896
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
        self.u0 = np.asarray([0 if x is None else x for x in e['u0']], dtype=float)
        self.c0 = np.asarray([0 if x is None else x for x in e['c0']], dtype=float)
        self.tau0 = np.asarray([1 if x is None else x for x in e['tau0']], dtype=float)
        self.v = np.asarray(s['commodities']['v'], dtype=float)
        self.pool = s['commodities']['pool']
        self.stocks = [tuple(map(int, p)) for p in self.l['stock_slots']]
        self.supplies = [tuple(map(int, p)) for p in self.l['supply_slots']]
        self.demands = [tuple(map(int, p)) for p in self.l['demands']]
        self.di = {p: i for i, p in enumerate(self.demands)}
        self.cp = {int(n): i for i, n in enumerate(self.l['chokepoints'])}
        pi = {(int(n), int(k)): float(x) for n, k, x in zip(s['sinks']['node'], s['sinks']['k'], s['sinks']['pi'])}
        self.pi = np.asarray([pi.get(p, 1) for p in self.demands])
        self.maxpi = max(1., float(np.max(self.pi)) if len(self.pi) else 1.)
        self.mem = {}
        self.routes = []
        groups = {}
        a = s['action_slots']
        for i, (edge, k, lane) in enumerate(zip(a['edge'], a['k'], a['lane'])):
            edge, k = int(edge), int(k)
            lane = -1 if lane is None else int(lane)
            es = [edge] if lane < 0 else list(map(int, s['lanes']['edges'][lane]))
            cps = [] if lane < 0 else list(map(int, s['lanes']['chokepoints'][lane]))
            if not es:
                continue
            src, dst = int(self.tail[es[0]]), int(self.head[es[-1]])
            alt = e['alt_of'][edge] if lane < 0 else s['lanes']['alt_of'][lane]
            self.routes.append((i, k, es, cps, src, dst))
            groups.setdefault((src, k, dst), []).append((alt, min(self.u0[x] for x in es)))
        inc, out = {}, {}
        for (src, k, dst), choices in groups.items():
            base = [q for alt, q in choices if alt is None]
            q = sum(base) if base else max(q for _, q in choices)
            inc[(dst, k)] = inc.get((dst, k), 0.) + q
            out[(src, k)] = out.get((src, k), 0.) + q
        self.rate = {}
        for p in set(inc) | set(out):
            self.rate[p] = min(inc[p], out[p]) if p in inc and p in out else max(inc.get(p, 0.), out.get(p, 0.))
        self.priority = {p: float(self.pi[i]) for p, i in self.di.items()}
        self.distance = {p: 0. for p in self.di}
        for _ in range(len(s['nodes']['id'])):
            changed = False
            for _, k, es, _, src, dst in self.routes:
                p, q = (src, k), (dst, k)
                if q in self.distance:
                    d = self.distance[q] + sum(self.tau0[x] for x in es) + 1
                    if d < self.distance.get(p, np.inf):
                        self.distance[p] = d
                        changed = True
                    self.priority[p] = max(self.priority.get(p, 0.), self.priority.get(q, 0.) / 1.12)
            if not changed:
                break

    def field(self, o, name, fallback, remember=False):
        old = self.mem.get(name, np.asarray(fallback)) if remember else np.asarray(fallback)
        x = np.asarray(o.get(name, old))
        m = o.get(name + '.observed')
        if m is not None and np.shape(m) == x.shape:
            x = np.where(m, x, old)
        if remember:
            self.mem[name] = x.copy()
        return x

    def act(self, o):
        t = int(o['week'][0])
        ne, nk = len(self.tail), len(self.v)
        ns = len(self.s['action_slots']['edge'])
        u = self.field(o, 'graph_now.u', self.u0, True)
        c = self.field(o, 'graph_now.c', self.c0, True)
        tau = np.maximum(1., self.field(o, 'graph_now.tau', self.tau0, True))
        tariff = self.field(o, 'graph_now.tariff', np.zeros((ne, nk)), True)
        banned = self.field(o, 'graph_now.prohibited', np.zeros((ne, nk)), True).astype(bool)
        opened = self.field(o, 'graph_now.open', np.ones(len(self.cp)), True)
        mask = np.asarray(o.get('action_mask', np.ones(ns)))
        available, onhand, future = {}, {}, {}
        for p, q in zip(self.stocks, self.field(o, 'stock.qty', np.zeros(len(self.stocks)))):
            available[p] = onhand[p] = max(0., float(q))
        for p, q in zip(self.supplies, self.field(o, 'graph_now.supply.avail', np.zeros(len(self.supplies)), True)):
            available[p] = available.get(p, 0.) + max(0., float(q))
        def add(p, w, q):
            if q > 0:
                future.setdefault(p, []).append((w, float(q)))
        def live(name):
            x = np.asarray(o.get(name, [])).reshape(-1)
            return np.flatnonzero(np.asarray(o.get(name + '.observed', np.ones(x.size)), dtype=bool)), x
        end = {}
        ix, x = live('closure_end.chokepoint')
        ew = np.asarray(o.get('closure_end.end_week', np.zeros(x.size)))
        em = np.asarray(o.get('closure_end.end_week.observed', np.zeros(x.size)))
        for j in ix:
            if j < ew.size and em[j]:
                end[int(x[j])] = int(ew[j]) + 1
        lanes = self.s['lanes']['edges']
        def destination(edge, lane, arrival, k):
            es = list(map(int, lanes[lane])) if 0 <= lane < len(lanes) else [edge]
            pos = es.index(edge) if edge in es else 0
            for ee in es[pos + 1:]:
                node = int(self.tail[ee])
                ci = self.cp.get(node)
                if ci is not None and opened[ci] < .01:
                    arrival = max(arrival, end.get(node, t + 4))
                if banned[ee, k]:
                    arrival = self.T + 2
                arrival += tau[ee]
            return int(self.head[es[-1]]), arrival
        ix, pq = live('pipeline.qty')
        for j in ix:
            edge, k = int(o['pipeline.edge'][j]), int(o['pipeline.k'][j])
            if not 0 <= edge < ne:
                continue
            lm = o.get('pipeline.lane.observed', np.ones(pq.size))
            lane = int(o['pipeline.lane'][j]) if lm[j] else -1
            dst, arrival = destination(edge, lane, max(t, int(o['pipeline.arrival_week'][j])), k)
            add((dst, k), arrival, pq[j])
        ix, wq = live('wip.qty')
        for j in ix:
            add((int(o['wip.node'][j]), int(o['wip.k'][j])), max(t, int(o['wip.out_week'][j])), wq[j])
        lots = np.asarray(o.get('queue_lots.qty', np.zeros((0, 0))))
        lm = np.asarray(o.get('queue_lots.qty.observed', np.ones(lots.shape)))
        for row, key in enumerate(self.l.get('lot_keys', [])[:lots.shape[0]]):
            cp, k, lane, edge = key
            cp, k, edge = int(cp), int(k), int(edge)
            release = t
            ci = self.cp.get(cp)
            if ci is not None and opened[ci] < .01:
                release = max(t, end.get(cp, t + 4))
            if banned[edge, k]:
                release = self.T + 2
            dst, arrival = destination(edge, -1 if lane is None else int(lane), release + tau[edge], k)
            add((dst, k), arrival, np.sum(np.where(lm[row], lots[row], 0.)))
        pending = {}
        ix, pe = live('pending_prohibitions.edge')
        for j in ix:
            p = (int(pe[j]), int(o['pending_prohibitions.k'][j]))
            pending[p] = min(pending.get(p, self.T + 1), int(o['pending_prohibitions.effective_week'][j]))
        fc = self.field(o, 'demand_forecast.qty', np.zeros((len(self.demands), 1)))
        backlog = self.field(o, 'backlog.qty', np.zeros(len(self.demands)))
        warning = self.field(o, 'warning.score', np.zeros(len(self.l.get('warning_units', []))))
        wm = {tuple(p): float(np.clip(q, 0, 1)) for p, q in zip(self.l.get('warning_units', []), warning)}
        def coverage(p, cutoff):
            return onhand.get(p, 0.) + sum(q for w, q in future.get(p, ()) if w <= cutoff)
        def need(r, lead):
            _, k, _, _, _, dst = r
            p = (dst, k)
            if p in self.di:
                di = self.di[p]
                h = min(max(1, self.T - t + 1), int(np.ceil(lead)) + 2)
                row = fc[di]
                n = min(h, len(row))
                target = max(0., float(backlog[di])) + float(np.sum(row[:n]))
                if h > n and n:
                    target += (h - n) * float(np.mean(row[-min(3, n):]))
                return max(0., target - coverage(p, t + h - 1))
            remaining = max(0., self.T - t - self.distance.get(p, 2.))
            duration = min(lead + 2., remaining)
            return max(0., self.rate.get(p, 0.) * duration - coverage(p, t + int(np.ceil(lead + 2.))))
        edge_left = np.maximum(0., u.copy())
        cp_left = {}
        for pool in ('tb', 'ct'):
            caps = self.field(o, 'graph_now.kappa.' + pool, np.full(len(self.cp), np.inf), True)
            for cp, ci in self.cp.items():
                cp_left[(cp, pool)] = max(0., float(caps[ci]) * float(opened[ci]))
        candidates = []
        for r in self.routes:
            i, k, es, cps, src, dst = r
            if not mask[i] or any(banned[e, k] for e in es):
                continue
            lead = sum(tau[e] for e in es)
            if t + lead > self.T or need(r, lead) <= 0:
                continue
            elapsed = 0.
            illegal = False
            for e in es:
                if pending.get((e, k), self.T + 1) <= t + elapsed:
                    illegal = True
                elapsed += tau[e]
            if illegal:
                continue
            cost = sum(c[e] + tariff[e, k] * self.v[k] for e in es)
            priority = max(1., self.priority.get((dst, k), self.maxpi * .25))
            risk = max([wm.get(('chokepoint', cp), 0.) for cp in cps] or [0.])
            score = (cost + .002 * self.v[k] * lead) / priority + .08 * risk
            candidates.append((score, lead, i, r))
        flows = np.zeros(ns)
        for _, lead, i, r in sorted(candidates):
            _, k, es, cps, src, dst = r
            p = (src, k)
            q = min(need(r, lead), available.get(p, 0.), min(edge_left[e] for e in es))
            for cp in cps:
                q = min(q, cp_left.get((cp, self.pool[k]), 0.))
            if q <= 0 or not np.isfinite(q):
                continue
            flows[i] = q
            available[p] -= q
            onhand[p] = max(0., onhand.get(p, 0.) - q)
            for e in es:
                edge_left[e] = max(0., edge_left[e] - q)
            for cp in cps:
                key = (cp, self.pool[k])
                cp_left[key] = max(0., cp_left[key] - q)
            add((dst, k), t + int(np.ceil(lead)), q)
        return {'flows': flows}
