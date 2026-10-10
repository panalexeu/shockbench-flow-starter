# -0.13205349354121276
import numpy as np

class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.T = int(config['T'])
        e, a = self.s['edges'], self.s['action_slots']
        self.tail = np.asarray(e['tail'], dtype=int)
        self.head = np.asarray(e['head'], dtype=int)
        self.u0 = np.asarray([0.0 if x is None else float(x) for x in e['u0']])
        self.c0 = np.asarray(e['c0'], dtype=float)
        self.tau0 = np.asarray(e['tau0'], dtype=float)
        self.v = np.asarray(self.s['commodities']['v'], dtype=float)
        self.names = self.s['commodities']['id']
        self.pools = self.s['commodities']['pool']
        self.types = self.s['nodes']['type']
        self.n = len(a['edge'])
        self.stockkeys = [tuple(map(int, x)) for x in self.l['stock_slots']]
        self.supplykeys = [tuple(map(int, x)) for x in self.l['supply_slots']]
        self.demands = [tuple(map(int, x)) for x in self.l['demands']]
        self.di = {key: j for j, key in enumerate(self.demands)}
        self.cpi = {int(n): j for j, n in enumerate(self.l['chokepoints'])}
        self.routes, self.groups = [], {}
        self.lanes = [list(map(int, p)) for p in self.s['lanes']['edges']]
        self.ldest = {j: int(self.head[p[-1]]) for j, p in enumerate(self.lanes) if p}
        for i, (edge, k, lane) in enumerate(zip(a['edge'], a['k'], a['lane'])):
            k, edge = int(k), int(edge)
            lane = -1 if lane is None else int(lane)
            path = [edge] if lane < 0 else self.lanes[lane]
            origin, dest = int(self.tail[path[0]]), int(self.head[path[-1]])
            cps = [int(self.head[x]) for x in path if int(self.head[x]) in self.cpi]
            self.routes.append((origin, dest, k, lane, path, cps))
            self.groups.setdefault((dest, k), []).append(i)
        self.penalty = {(int(n), int(k)): float(p) for n, k, p in zip(self.s['sinks']['node'], self.s['sinks']['k'], self.s['sinks']['pi'])}
        self.cache = {}
        self.rate = None
        self.previous_position = None
        self.base = {}
        self.safety = {}
        self.initial_demand = {}
        # Approximate downstream transport and processing time, without assuming node indices.
        self.downstream = {int(n): 0.0 for n, k in self.demands}
        for _ in range(len(self.types)):
            changed = False
            for origin, dest, k, lane, path, cps in self.routes:
                if dest not in self.downstream:
                    continue
                processing = 1.5 if self.types[origin] in ('fab', 'osat') else 0.0
                d = sum(max(0.0, self.tau0[x]) for x in path) + processing + self.downstream[dest]
                if d < self.downstream.get(origin, float('inf')):
                    self.downstream[origin] = d
                    changed = True
            if not changed:
                break

    def read(self, o, key, default):
        x = np.asarray(o.get(key, default)).copy()
        mask = o.get(key + '.observed')
        if mask is not None and np.shape(mask) == x.shape:
            x = np.where(np.asarray(mask, dtype=bool), x, self.cache.get(key, np.asarray(default)))
        self.cache[key] = x.copy()
        return x

    def live(self, o, key):
        q = np.asarray(o.get(key, []))
        mask = np.asarray(o.get(key + '.observed', np.ones(q.shape)), dtype=bool)
        return np.flatnonzero(mask & (q > 0))

    def initialize(self, o, forecast):
        rates = {}
        q = np.asarray(o.get('pipeline.qty', []))
        ed = np.asarray(o.get('pipeline.edge', []), dtype=int)
        kk = np.asarray(o.get('pipeline.k', []), dtype=int)
        ll = np.asarray(o.get('pipeline.lane', np.full(len(q), -1)), dtype=int)
        lm = np.asarray(o.get('pipeline.lane.observed', np.ones(len(q))), dtype=bool)
        for j in self.live(o, 'pipeline.qty'):
            e, k = int(ed[j]), int(kk[j])
            if 0 <= e < len(self.head) and 0 <= k < len(self.v):
                lane = int(ll[j]) if lm[j] else -1
                key = (lane, e, k)
                rates[key] = rates.get(key, 0.0) + float(q[j]) / max(1.0, self.tau0[e])
        incoming, outgoing = {}, {}
        for origin, dest, k, lane, path, cps in self.routes:
            vals = [rates.get((lane, e, k), 0.0) for e in path]
            vals = [x for x in vals if x > 0]
            r = float(np.median(vals)) if vals else 0.0
            incoming[(dest, k)] = incoming.get((dest, k), 0.0) + r
            outgoing[(origin, k)] = outgoing.get((origin, k), 0.0) + r
        self.rate = {}
        for key, ids in self.groups.items():
            r = max(incoming.get(key, 0.0), outgoing.get(key, 0.0))
            if r <= 0:
                r = 0.12 * max(self.u0[self.routes[i][4][0]] for i in ids)
            self.rate[key] = r
        for key, j in self.di.items():
            r = max(0.0, float(np.mean(forecast[j])))
            self.rate[key] = r
            self.initial_demand[key] = max(r, 1e-8)
        self.base = self.rate.copy()
        stock = np.asarray(o.get('stock.qty', np.zeros(len(self.stockkeys))))
        for j, key in enumerate(self.stockkeys):
            r = self.rate.get(key, 0.0)
            self.safety[key] = min(max(0.0, float(stock[j])), 3.0 * r)

    def act(self, observation):
        o = observation
        t = int(np.asarray(o['week']).flat[0])
        forecast = self.read(o, 'demand_forecast.qty', np.zeros((len(self.demands), 1)))
        if self.rate is None:
            self.initialize(o, forecast)
        u = np.maximum(0.0, self.read(o, 'graph_now.u', self.u0).astype(float))
        c = self.read(o, 'graph_now.c', self.c0)
        tau = np.maximum(0.0, self.read(o, 'graph_now.tau', self.tau0))
        prohibited = self.read(o, 'graph_now.prohibited', np.zeros((len(u), len(self.v)), dtype=np.int8))
        tariff = self.read(o, 'graph_now.tariff', np.zeros((len(u), len(self.v))))
        opened = self.read(o, 'graph_now.open', np.ones(len(self.cpi)))
        ktb = self.read(o, 'graph_now.kappa.tb', np.full(len(self.cpi), np.inf))
        kct = self.read(o, 'graph_now.kappa.ct', np.full(len(self.cpi), np.inf))
        mask = np.asarray(o.get('action_mask', np.ones(self.n)), dtype=bool)
        backlog = self.read(o, 'backlog.qty', np.zeros(len(self.demands)))
        stockq = self.read(o, 'stock.qty', np.zeros(len(self.stockkeys)))
        stock = {key: max(0.0, float(stockq[j])) for j, key in enumerate(self.stockkeys)}
        available = stock.copy()
        supply = self.read(o, 'graph_now.supply.avail', np.zeros(len(self.supplykeys)))
        for j, key in enumerate(self.supplykeys):
            available[key] = available.get(key, 0.0) + max(0.0, float(supply[j]))
        ends, pending = {}, {}
        ce = np.asarray(o.get('closure_end.end_week', []))
        cm = np.asarray(o.get('closure_end.end_week.observed', np.zeros(len(ce))), dtype=bool)
        cn = np.asarray(o.get('closure_end.chokepoint', []), dtype=int)
        for j in np.flatnonzero(cm):
            ends[int(cn[j])] = max(ends.get(int(cn[j]), t), int(ce[j]))
        pp = np.asarray(o.get('pending_prohibitions.effective_week', []))
        pm = np.asarray(o.get('pending_prohibitions.effective_week.observed', np.zeros(len(pp))), dtype=bool)
        pe = np.asarray(o.get('pending_prohibitions.edge', []), dtype=int)
        pk = np.asarray(o.get('pending_prohibitions.k', []), dtype=int)
        for j in np.flatnonzero(pm):
            key = (int(pe[j]), int(pk[j]))
            pending[key] = min(pending.get(key, self.T + 1), int(pp[j]))
        lots = np.asarray(o.get('queue_lots.qty', []))
        queue = {}
        for j, entry in enumerate(self.l.get('lot_keys', [])):
            if j >= len(lots):
                break
            cp, k, lane, edge = map(int, entry)
            key = (cp, self.pools[k])
            queue[key] = queue.get(key, 0.0) + max(0.0, float(np.sum(lots[j])))
        poolcap = {}
        for cp, j in self.cpi.items():
            for pool, caps in [('tb', ktb), ('ct', kct)]:
                poolcap[(cp, pool)] = max(0.0, float(caps[j]) * float(opened[j]))

        def delay(cp, k, when):
            if cp not in self.cpi:
                return 0.0
            j = self.cpi[cp]
            op = float(opened[j])
            wait = max(0.0, ends.get(cp, t + 6) + 1 - when) if op < 0.05 else 0.0
            cap = poolcap.get((cp, self.pools[k]), np.inf)
            if cap > 0:
                wait += min(8.0, queue.get((cp, self.pools[k]), 0.0) / cap)
            elif op < 0.05:
                wait += 0.5
            return wait

        def arrival(path, k, start, check=False):
            when = float(start)
            for e in path:
                cp = int(self.tail[e])
                when += delay(cp, k, when)
                if check and pending.get((e, k), self.T + 1) <= when:
                    return float('inf')
                when += tau[e]
            return when

        events = {}
        total = stock.copy()
        def add(key, qty, eta):
            if qty <= 0:
                return
            events.setdefault(key, []).append((float(eta), float(qty)))
            total[key] = total.get(key, 0.0) + float(qty)
        pq = np.asarray(o.get('pipeline.qty', []))
        ped = np.asarray(o.get('pipeline.edge', []), dtype=int)
        pkk = np.asarray(o.get('pipeline.k', []), dtype=int)
        pll = np.asarray(o.get('pipeline.lane', np.full(len(pq), -1)), dtype=int)
        plm = np.asarray(o.get('pipeline.lane.observed', np.ones(len(pq))), dtype=bool)
        pa = np.asarray(o.get('pipeline.arrival_week', np.full(len(pq), t + 1)))
        for j in self.live(o, 'pipeline.qty'):
            e, k = int(ped[j]), int(pkk[j])
            if not 0 <= e < len(self.head):
                continue
            lane = int(pll[j]) if plm[j] else -1
            dest, eta = int(self.head[e]), float(pa[j])
            if lane in self.ldest and e in self.lanes[lane]:
                path = self.lanes[lane]
                dest = self.ldest[lane]
                eta = arrival(path[path.index(e) + 1:], k, max(t, eta))
            key = (dest, k)
            add(key, float(pq[j]), eta)
            if pa[j] <= t and int(self.head[e]) == dest:
                available[key] = available.get(key, 0.0) + float(pq[j])
        for j, entry in enumerate(self.l.get('lot_keys', [])):
            if j >= len(lots):
                break
            cp, k, lane, e = map(int, entry)
            path = self.lanes[lane] if lane in self.ldest else [e]
            if e in path:
                path = path[path.index(e):]
            dest = self.ldest.get(lane, int(self.head[e]))
            add((dest, k), max(0.0, float(np.sum(lots[j]))), arrival(path, k, t))
        wn = np.asarray(o.get('wip.node', []), dtype=int)
        wk = np.asarray(o.get('wip.k', []), dtype=int)
        wq = np.asarray(o.get('wip.qty', []))
        ww = np.asarray(o.get('wip.out_week', np.full(len(wq), t + 1)))
        for j in self.live(o, 'wip.qty'):
            key = (int(wn[j]), int(wk[j]))
            add(key, float(wq[j]), float(ww[j]))
            if ww[j] <= t:
                available[key] = available.get(key, 0.0) + float(wq[j])
        # Conservation of inventory position estimates depletion, including downstream dispatch.
        if self.previous_position is not None:
            executed = np.asarray(o.get('last_week.clip.executed', np.zeros(self.n)))
            incoming = {}
            for i, (_, dest, k, _, _, _) in enumerate(self.routes):
                key = (dest, k)
                incoming[key] = incoming.get(key, 0.0) + max(0.0, float(executed[i]))
            for key in self.rate:
                if key in self.di or key not in self.previous_position:
                    continue
                depletion = self.previous_position[key] + incoming.get(key, 0.0) - total.get(key, 0.0)
                base = self.base[key]
                if base > 0 and depletion >= 0 and depletion < 4.0 * base:
                    measured = float(np.clip(depletion, 0.45 * base, 1.8 * base))
                    self.rate[key] = 0.88 * self.rate[key] + 0.12 * measured
        self.previous_position = total.copy()
        ratios = [float(np.mean(forecast[j])) / self.initial_demand[key] for key, j in self.di.items() if 'chip' in str(self.names[key[1]])]
        scale = float(np.clip(np.mean(ratios), 0.65, 1.5)) if ratios else 1.0
        flows = np.zeros(self.n)
        remaining = u.copy()
        cpused = {}
        requests = []
        for key, ids in self.groups.items():
            dest, k = key
            rate = self.rate[key]
            if key in self.di:
                j = self.di[key]
                rate = max(0.0, float(np.mean(forecast[j])))
                safety = 1.25 * rate
                priority = self.penalty.get(key, self.v[k])
                last = self.T
                debt = max(0.0, float(backlog[j]))
            else:
                if 'chip' in str(self.names[k]) or 'wafer' in str(self.names[k]):
                    rate *= scale
                safety = max(0.6 * rate, self.safety.get(key, rate))
                priority = max(10.0, self.v[k])
                last = self.T - self.downstream.get(dest, 1.0)
                debt = 0.0
            options = []
            cover = max(0.0, stock.get(key, 0.0) - debt) / max(rate, 1e-8)
            for i in ids:
                origin, _, _, lane, path, cps = self.routes[i]
                if not mask[i] or any(prohibited[e, k] for e in path) or remaining[path[0]] <= 0:
                    continue
                eta = arrival(path, k, t, True)
                if eta > last or not np.isfinite(eta):
                    continue
                lead = max(1.0, eta - t)
                freight = sum(float(c[e]) + max(0.0, float(tariff[e, k])) * self.v[k] for e in path)
                imminent = sum(q for w, q in events.get(key, []) if w <= t + cover + 1)
                effective_cover = cover + imminent / max(rate, 1e-8)
                shortage_cost = priority * min(1.0, max(0.0, lead - effective_cover) / max(lead, 1.0))
                options.append((freight + 0.65 * shortage_cost, i, lead, eta))
            if options:
                options.sort()
                lead = options[0][2]
                horizon = min(lead + safety / max(rate, 1e-8), max(0.0, last - t + 1))
                pos = stock.get(key, 0.0) + sum(q for w, q in events.get(key, []) if w <= t + horizon)
                need = max(0.0, rate * horizon + debt - pos)
                if need > 1e-8:
                    requests.append((priority * (1.0 + min(4.0, need / max(rate, 1.0))), key, rate, safety, last, debt, options))
        requests.sort(key=lambda x: x[0], reverse=True)
        for _, key, rate, safety, last, debt, options in requests:
            for score, i, lead, eta in options:
                origin, dest, k, lane, path, cps = self.routes[i]
                horizon = min(lead + safety / max(rate, 1e-8), max(0.0, last - t + 1))
                pos = stock.get(key, 0.0) + sum(q for w, q in events.get(key, []) if w <= t + horizon)
                need = max(0.0, rate * horizon + debt - pos)
                source = (origin, k)
                inventory = available.get(source, 0.0)
                if self.types[origin] == 'source' and source not in available:
                    inventory = remaining[path[0]]
                cap = remaining[path[0]]
                for cp in cps:
                    poolkey = (cp, self.pools[k])
                    pc = poolcap.get(poolkey, np.inf)
                    # A closed route can still be useful if its reopening is announced.
                    if pc > 0:
                        cap = min(cap, max(0.0, pc - cpused.get(poolkey, 0.0)))
                q = min(need, inventory, cap)
                if q <= 1e-8:
                    continue
                flows[i] += q
                remaining[path[0]] -= q
                available[source] = max(0.0, inventory - q)
                events.setdefault(key, []).append((eta, q))
                for cp in cps:
                    poolkey = (cp, self.pools[k])
                    cpused[poolkey] = cpused.get(poolkey, 0.0) + q
        return {'flows': np.maximum(0.0, np.nan_to_num(flows, nan=0.0, posinf=0.0, neginf=0.0))}
