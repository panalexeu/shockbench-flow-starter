# -0.08906010618141873
import numpy as np

class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.T = int(config['T'])
        e = self.s['edges']
        a = self.s['action_slots']
        self.tail = np.asarray(e['tail'], dtype=int)
        self.head = np.asarray(e['head'], dtype=int)
        self.u0 = np.asarray([0.0 if x is None else float(x) for x in e['u0']])
        self.c0 = np.asarray(e['c0'], dtype=float)
        self.tau0 = np.asarray(e['tau0'], dtype=float)
        self.v = np.asarray(self.s['commodities']['v'], dtype=float)
        self.pool = self.s['commodities']['pool']
        self.names = self.s['commodities']['id']
        self.types = self.s['nodes']['type']
        self.stockkeys = [tuple(map(int, x)) for x in self.l['stock_slots']]
        self.supplykeys = [tuple(map(int, x)) for x in self.l['supply_slots']]
        self.demands = [tuple(map(int, x)) for x in self.l['demands']]
        self.di = {x: j for j, x in enumerate(self.demands)}
        self.cp = {int(n): j for j, n in enumerate(self.l['chokepoints'])}
        self.lanes = [list(map(int, x)) for x in self.s['lanes']['edges']]
        self.routes = []
        self.groups = {}
        self.outgoing = {}
        for i, (ed, k, lane) in enumerate(zip(a['edge'], a['k'], a['lane'])):
            ed, k = int(ed), int(k)
            lane = -1 if lane is None else int(lane)
            path = self.lanes[lane] if lane >= 0 else [ed]
            origin, dest = int(self.tail[path[0]]), int(self.head[path[-1]])
            self.routes.append((origin, dest, k, lane, path))
            self.groups.setdefault((dest, k), []).append(i)
            self.outgoing.setdefault((origin, k), []).append(i)
        self.n = len(self.routes)
        self.penalty = {(int(n), int(k)): float(p) for n, k, p in zip(self.s['sinks']['node'], self.s['sinks']['k'], self.s['sinks']['pi'])}
        self.downstream = {n: 0.0 for n, k in self.demands}
        for _ in range(len(self.types)):
            changed = False
            for origin, dest, k, lane, path in self.routes:
                if dest not in self.downstream:
                    continue
                processing = 2.0 if self.types[origin] in ('fab', 'osat') else 0.0
                length = sum(self.tau0[x] for x in path) + processing + self.downstream[dest]
                if length < self.downstream.get(origin, float('inf')):
                    self.downstream[origin] = length
                    changed = True
            if not changed:
                break
        self.cache = {}
        self.rate = None
        self.prev = None
        self.demand0 = {}
        self.base = {}

    def read(self, o, key, default):
        x = np.asarray(o.get(key, default)).copy()
        m = o.get(key + '.observed')
        if m is not None and np.shape(m) == x.shape:
            x = np.where(np.asarray(m, dtype=bool), x, self.cache.get(key, np.asarray(default)))
        self.cache[key] = x.copy()
        return x

    def live(self, o, key):
        x = np.asarray(o.get(key, []))
        m = np.asarray(o.get(key + '.observed', np.ones(x.shape)), dtype=bool)
        return np.flatnonzero(m & (x > 0))

    def initialize(self, o, forecast):
        rates = {}
        qty = np.asarray(o.get('pipeline.qty', []))
        edges = np.asarray(o.get('pipeline.edge', []), dtype=int)
        goods = np.asarray(o.get('pipeline.k', []), dtype=int)
        lanes = np.asarray(o.get('pipeline.lane', np.full(len(qty), -1)), dtype=int)
        lm = np.asarray(o.get('pipeline.lane.observed', np.ones(len(qty))), dtype=bool)
        for j in self.live(o, 'pipeline.qty'):
            e, k = int(edges[j]), int(goods[j])
            if not 0 <= e < len(self.head):
                continue
            lane = int(lanes[j]) if lm[j] else -1
            key = (lane, e, k)
            rates[key] = rates.get(key, 0.0) + float(qty[j]) / max(1.0, self.tau0[e])
        incoming, outgoing = {}, {}
        for origin, dest, k, lane, path in self.routes:
            values = [rates.get((lane, ed, k), 0.0) for ed in path]
            positive = [x for x in values if x > 0]
            r = float(np.median(positive)) if positive else 0.0
            incoming[(dest, k)] = incoming.get((dest, k), 0.0) + r
            outgoing[(origin, k)] = outgoing.get((origin, k), 0.0) + r
        self.rate = {}
        for key, ids in self.groups.items():
            r = max(incoming.get(key, 0.0), outgoing.get(key, 0.0))
            if r <= 0:
                r = 0.45 * max(self.u0[self.routes[i][4][0]] for i in ids)
            self.rate[key] = r
        for key, j in self.di.items():
            r = max(0.0, float(np.mean(forecast[j])))
            self.rate[key] = r
            self.demand0[key] = max(r, 1e-8)
        self.base = self.rate.copy()

    def act(self, observation):
        o = observation
        t = int(np.asarray(o['week']).flat[0])
        forecast = self.read(o, 'demand_forecast.qty', np.zeros((len(self.demands), 1)))
        if self.rate is None:
            self.initialize(o, forecast)
        u = np.maximum(0.0, self.read(o, 'graph_now.u', self.u0))
        c = self.read(o, 'graph_now.c', self.c0)
        tau = np.maximum(0.0, self.read(o, 'graph_now.tau', self.tau0))
        prohibited = self.read(o, 'graph_now.prohibited', np.zeros((len(u), len(self.v)), dtype=np.int8))
        tariff = self.read(o, 'graph_now.tariff', np.zeros((len(u), len(self.v))))
        opened = self.read(o, 'graph_now.open', np.ones(len(self.cp)))
        ktb = self.read(o, 'graph_now.kappa.tb', np.full(len(self.cp), np.inf))
        kct = self.read(o, 'graph_now.kappa.ct', np.full(len(self.cp), np.inf))
        mask = np.asarray(o.get('action_mask', np.ones(self.n)), dtype=bool)
        stockq = self.read(o, 'stock.qty', np.zeros(len(self.stockkeys)))
        stock = {key: max(0.0, float(stockq[j])) for j, key in enumerate(self.stockkeys)}
        available = stock.copy()
        supply = self.read(o, 'graph_now.supply.avail', np.zeros(len(self.supplykeys)))
        for j, key in enumerate(self.supplykeys):
            available[key] = available.get(key, 0.0) + max(0.0, float(supply[j]))
        backlog = self.read(o, 'backlog.qty', np.zeros(len(self.demands)))
        ends, pending = {}, {}
        for j in self.live(o, 'closure_end.end_week'):
            node = int(o['closure_end.chokepoint'][j])
            ends[node] = max(ends.get(node, t), int(o['closure_end.end_week'][j]))
        for j in self.live(o, 'pending_prohibitions.effective_week'):
            key = (int(o['pending_prohibitions.edge'][j]), int(o['pending_prohibitions.k'][j]))
            pending[key] = min(pending.get(key, self.T + 1), int(o['pending_prohibitions.effective_week'][j]))
        lots = np.asarray(o.get('queue_lots.qty', []))
        queue = {}
        for j, entry in enumerate(self.l.get('lot_keys', [])):
            if j >= len(lots):
                break
            cp, k, lane, ed = map(int, entry)
            key = (cp, self.pool[k])
            queue[key] = queue.get(key, 0.0) + max(0.0, float(np.sum(lots[j])))

        def arrival(path, k, start, check=False):
            when = float(start)
            for ed in path:
                node = int(self.tail[ed])
                if node in self.cp:
                    j = self.cp[node]
                    op = float(opened[j])
                    if op < 0.05:
                        when = max(when, ends.get(node, t + 8) + 1)
                        cap = float((ktb if self.pool[k] == 'tb' else kct)[j])
                    else:
                        cap = float((ktb if self.pool[k] == 'tb' else kct)[j]) * op
                    if cap > 0:
                        when += min(10.0, queue.get((node, self.pool[k]), 0.0) / cap)
                    else:
                        when += 8.0
                if check and pending.get((ed, k), self.T + 1) <= when:
                    return float('inf')
                when += tau[ed]
            return when

        events = {}
        total = stock.copy()
        def add(key, quantity, when):
            if quantity > 0:
                events.setdefault(key, []).append((float(when), float(quantity)))
                total[key] = total.get(key, 0.0) + float(quantity)
        pq = np.asarray(o.get('pipeline.qty', []))
        pe = np.asarray(o.get('pipeline.edge', []), dtype=int)
        pk = np.asarray(o.get('pipeline.k', []), dtype=int)
        pl = np.asarray(o.get('pipeline.lane', np.full(len(pq), -1)), dtype=int)
        lm = np.asarray(o.get('pipeline.lane.observed', np.ones(len(pq))), dtype=bool)
        pa = np.asarray(o.get('pipeline.arrival_week', np.full(len(pq), t + 1)))
        for j in self.live(o, 'pipeline.qty'):
            ed, k = int(pe[j]), int(pk[j])
            if not 0 <= ed < len(self.head):
                continue
            lane = int(pl[j]) if lm[j] else -1
            dest, when = int(self.head[ed]), float(pa[j])
            if 0 <= lane < len(self.lanes) and ed in self.lanes[lane]:
                path = self.lanes[lane]
                dest = int(self.head[path[-1]])
                when = arrival(path[path.index(ed) + 1:], k, max(t, when))
            key = (dest, k)
            add(key, float(pq[j]), when)
            if pa[j] <= t and dest == int(self.head[ed]):
                available[key] = available.get(key, 0.0) + float(pq[j])
        for j, entry in enumerate(self.l.get('lot_keys', [])):
            if j >= len(lots):
                break
            cp, k, lane, ed = map(int, entry)
            path = self.lanes[lane] if 0 <= lane < len(self.lanes) else [ed]
            dest = int(self.head[path[-1]])
            path = path[path.index(ed):] if ed in path else [ed]
            add((dest, k), max(0.0, float(np.sum(lots[j]))), arrival(path, k, t))
        wq = np.asarray(o.get('wip.qty', []))
        for j in self.live(o, 'wip.qty'):
            key = (int(o['wip.node'][j]), int(o['wip.k'][j]))
            when = float(o['wip.out_week'][j])
            add(key, float(wq[j]), when)
            if when <= t:
                available[key] = available.get(key, 0.0) + float(wq[j])
        executed = np.maximum(0.0, np.asarray(o.get('last_week.clip.executed', np.zeros(self.n))))
        if self.prev is not None:
            incoming, outgoing = {}, {}
            for i, (origin, dest, k, lane, path) in enumerate(self.routes):
                incoming[(dest, k)] = incoming.get((dest, k), 0.0) + float(executed[i])
                outgoing[(origin, k)] = outgoing.get((origin, k), 0.0) + float(executed[i])
            for key in self.rate:
                if key in self.di:
                    continue
                depletion = self.prev.get(key, 0.0) + incoming.get(key, 0.0) - total.get(key, 0.0)
                measurement = max(0.0, depletion, outgoing.get(key, 0.0))
                ceiling = sum(self.u0[self.routes[i][4][0]] for i in self.groups[key])
                measurement = min(measurement, max(ceiling, self.base[key]))
                weight = 0.45 if measurement > self.rate[key] else 0.16
                self.rate[key] = (1.0 - weight) * self.rate[key] + weight * measurement
        self.prev = total.copy()
        chip_ratios = [float(np.mean(forecast[j])) / self.demand0[key] for key, j in self.di.items() if 'chip' in str(self.names[key[1]])]
        scale = float(np.clip(np.mean(chip_ratios), 0.7, 1.5)) if chip_ratios else 1.0
        requests = []
        for key, ids in self.groups.items():
            dest, k = key
            sink = key in self.di
            if sink:
                j = self.di[key]
                rate = max(0.0, float(np.mean(forecast[j])))
                debt = max(0.0, float(backlog[j]))
                priority = self.penalty.get(key, self.v[k])
                last = float(self.T)
                buffer = 1.5
            else:
                rate = max(self.rate[key], 0.15 * self.base[key])
                if 'chip' in str(self.names[k]) or 'wafer' in str(self.names[k]):
                    rate = max(rate, 0.65 * self.base[key] * scale)
                debt = 0.0
                priority = max(10.0, self.v[k])
                last = self.T - self.downstream.get(dest, 1.0)
                buffer = 2.25 if self.types[dest] in ('grid', 'fab', 'osat') else 1.5
            cover = max(0.0, stock.get(key, 0.0) - debt) / max(rate, 1e-8)
            options = []
            for i in ids:
                origin, dest, k, lane, path = self.routes[i]
                if not mask[i] or u[path[0]] <= 0 or any(prohibited[ed, k] for ed in path):
                    continue
                eta = arrival(path, k, t, True)
                if not np.isfinite(eta) or eta > last:
                    continue
                lead = max(1.0, eta - t)
                freight = sum(float(c[ed]) + max(0.0, float(tariff[ed, k])) * self.v[k] for ed in path)
                early = sum(q for w, q in events.get(key, []) if w <= t + cover + 1.0)
                coverage = cover + early / max(rate, 1e-8)
                urgency = max(0.0, lead - coverage)
                score = freight + 0.75 * priority * min(3.0, urgency)
                options.append((score, i, lead, eta))
            if not options or rate <= 0:
                continue
            options.sort()
            horizon = min(options[0][2] + buffer, max(0.0, last - t + 1))
            position = stock.get(key, 0.0) + sum(q for w, q in events.get(key, []) if w <= t + horizon)
            need = max(0.0, rate * horizon + debt - position)
            if need > 1e-8:
                requests.append((priority * (1.0 + min(5.0, need / max(rate, 1e-8))), key, rate, debt, buffer, last, options))
        requests.sort(key=lambda x: x[0], reverse=True)
        flows = np.zeros(self.n)
        remaining = u.copy()
        for priority, key, rate, debt, buffer, last, options in requests:
            for score, i, lead, eta in options:
                origin, dest, k, lane, path = self.routes[i]
                horizon = min(lead + buffer, max(0.0, last - t + 1))
                position = stock.get(key, 0.0) + sum(q for w, q in events.get(key, []) if w <= t + horizon)
                need = max(0.0, rate * horizon + debt - position)
                source = (origin, k)
                inventory = available.get(source, 0.0)
                if self.types[origin] == 'source' and source not in available:
                    inventory = remaining[path[0]]
                quantity = min(need, inventory, remaining[path[0]])
                if quantity <= 1e-8:
                    continue
                flows[i] += quantity
                remaining[path[0]] -= quantity
                available[source] = max(0.0, inventory - quantity)
                events.setdefault(key, []).append((eta, quantity))
                if quantity >= need - 1e-8:
                    break
        return {'flows': np.maximum(0.0, np.nan_to_num(flows, nan=0.0, posinf=0.0, neginf=0.0))}
