# 0.32033595082259564
import numpy as np

class Agent:
    def __init__(self, config):
        self.s, self.l = config['static'], config['layout']
        self.T = int(config['T'])
        e = self.s['edges']
        self.tail = np.asarray(e['tail'], dtype=int)
        self.head = np.asarray(e['head'], dtype=int)
        self.u0 = np.asarray([0.0 if x is None else float(x) for x in e['u0']])
        self.c0 = np.asarray(e['c0'], dtype=float)
        self.tau0 = np.asarray(e['tau0'], dtype=float)
        self.v = np.asarray(self.s['commodities']['v'], dtype=float)
        self.names = self.s['commodities']['id']
        self.pool = self.s['commodities']['pool']
        self.types = self.s['nodes']['type']
        self.stockkeys = [tuple(map(int, x)) for x in self.l['stock_slots']]
        self.supplykeys = [tuple(map(int, x)) for x in self.l['supply_slots']]
        self.demands = [tuple(map(int, x)) for x in self.l['demands']]
        self.di = {x: j for j, x in enumerate(self.demands)}
        self.cp = {int(n): j for j, n in enumerate(self.l['chokepoints'])}
        self.lanes = [list(map(int, p)) for p in self.s['lanes']['edges']]
        self.routes, self.groups = [], {}
        a = self.s['action_slots']
        for i, (ed, k, lane) in enumerate(zip(a['edge'], a['k'], a['lane'])):
            ed, k = int(ed), int(k)
            lane = -1 if lane is None else int(lane)
            path = self.lanes[lane] if lane >= 0 else [ed]
            origin, dest = int(self.tail[path[0]]), int(self.head[path[-1]])
            self.routes.append((origin, dest, k, lane, path))
            self.groups.setdefault((dest, k), []).append(i)
        self.n = len(self.routes)
        sk = self.s['sinks']
        self.penalty = {(int(n), int(k)): float(p) for n, k, p in zip(sk['node'], sk['k'], sk['pi'])}
        self.carry = {(int(n), int(k)): bool(b) for n, k, b in zip(sk['node'], sk['k'], sk['backlog'])}
        self.downstream = {n: 0.0 for n, k in self.demands}
        for _ in range(len(self.types)):
            changed = False
            for origin, dest, k, lane, path in self.routes:
                if dest not in self.downstream:
                    continue
                distance = sum(self.tau0[x] for x in path) + self.downstream[dest]
                if self.types[origin] in ('fab', 'osat'):
                    distance += 2.0
                if distance < self.downstream.get(origin, np.inf):
                    self.downstream[origin] = distance
                    changed = True
            if not changed:
                break
        self.release_pairs = [tuple(map(int, x)) for x in self.l.get('release_pairs', [])]
        self.release_index = {x: j for j, x in enumerate(self.release_pairs)}
        self.overrides = []
        z = self.s.get('override_slots', {})
        for node, k, ed, lane in zip(z.get('chokepoint', []), z.get('k', []), z.get('out_edge', []), z.get('lane', [])):
            node, k, ed = int(node), int(k), int(ed)
            lane = -1 if lane is None else int(lane)
            path = self.lanes[lane] if 0 <= lane < len(self.lanes) else [ed]
            path = path[path.index(ed):] if ed in path else [ed]
            self.overrides.append((node, k, ed, lane, path))
        self.cache = {}
        self.rate = None
        self.previous = None
        self.base = {}
        self.initial_demand = {}
        self.initial_safety = {}

    def read(self, o, key, default):
        x = np.asarray(o.get(key, default)).copy()
        m = o.get(key + '.observed')
        if m is not None and np.shape(m) == x.shape:
            old = self.cache.get(key, np.asarray(default))
            if old.shape != x.shape:
                try:
                    old = np.broadcast_to(default, x.shape)
                except ValueError:
                    old = np.zeros_like(x)
            x = np.where(np.asarray(m, dtype=bool), x, old)
        self.cache[key] = x.copy()
        return x

    def live(self, o, key):
        x = np.asarray(o.get(key, []))
        m = np.asarray(o.get(key + '.observed', np.ones(x.shape)), dtype=bool)
        return np.flatnonzero(m & np.isfinite(x) & (x > 0))

    def initialize(self, o, forecast, stock):
        rates = {}
        q = np.asarray(o.get('pipeline.qty', []))
        edges = np.asarray(o.get('pipeline.edge', []), dtype=int)
        goods = np.asarray(o.get('pipeline.k', []), dtype=int)
        lanes = np.asarray(o.get('pipeline.lane', np.full(len(q), -1)), dtype=int)
        lm = np.asarray(o.get('pipeline.lane.observed', np.ones(len(q))), dtype=bool)
        for j in self.live(o, 'pipeline.qty'):
            ed, k = int(edges[j]), int(goods[j])
            if not 0 <= ed < len(self.head):
                continue
            lane = int(lanes[j]) if lm[j] else -1
            key = (lane, ed, k)
            rates[key] = rates.get(key, 0.0) + float(q[j]) / max(1.0, self.tau0[ed])
        incoming, outgoing = {}, {}
        for origin, dest, k, lane, path in self.routes:
            vals = [rates.get((lane, ed, k), 0.0) for ed in path]
            vals = [x for x in vals if x > 0]
            r = float(np.median(vals)) if vals else 0.0
            incoming[(dest, k)] = incoming.get((dest, k), 0.0) + r
            outgoing[(origin, k)] = outgoing.get((origin, k), 0.0) + r
        self.rate = {}
        for key, ids in self.groups.items():
            r = max(incoming.get(key, 0.0), outgoing.get(key, 0.0))
            if r <= 0:
                r = 0.30 * max(self.u0[self.routes[i][4][0]] for i in ids)
            self.rate[key] = r
        for key, j in self.di.items():
            r = max(0.0, float(np.mean(forecast[j])))
            self.rate[key] = r
            self.initial_demand[key] = max(r, 1e-8)
        self.base = self.rate.copy()
        for key, r in self.rate.items():
            self.initial_safety[key] = min(3.0, stock.get(key, 0.0) / max(r, 1e-8))

    def act(self, observation):
        o = observation
        t = int(np.asarray(o['week']).flat[0])
        forecast = np.maximum(0.0, self.read(o, 'demand_forecast.qty', np.zeros((len(self.demands), 1))))
        stockq = self.read(o, 'stock.qty', np.zeros(len(self.stockkeys)))
        stock = {key: max(0.0, float(stockq[j])) for j, key in enumerate(self.stockkeys)}
        if self.rate is None:
            self.initialize(o, forecast, stock)
        u = np.maximum(0.0, self.read(o, 'graph_now.u', self.u0))
        c = self.read(o, 'graph_now.c', self.c0)
        tau = np.maximum(0.0, self.read(o, 'graph_now.tau', self.tau0))
        prohibited = self.read(o, 'graph_now.prohibited', np.zeros((len(u), len(self.v)), dtype=np.int8))
        tariff = self.read(o, 'graph_now.tariff', np.zeros((len(u), len(self.v))))
        opened = self.read(o, 'graph_now.open', np.ones(len(self.cp)))
        ktb = self.read(o, 'graph_now.kappa.tb', np.full(len(self.cp), np.inf))
        kct = self.read(o, 'graph_now.kappa.ct', np.full(len(self.cp), np.inf))
        mask = np.asarray(o.get('action_mask', np.ones(self.n)), dtype=bool)
        backlog = self.read(o, 'backlog.qty', np.zeros(len(self.demands)))
        available = stock.copy()
        supply = self.read(o, 'graph_now.supply.avail', np.zeros(len(self.supplykeys)))
        for j, key in enumerate(self.supplykeys):
            available[key] = available.get(key, 0.0) + max(0.0, float(supply[j]))
        ends, pending = {}, {}
        for j in self.live(o, 'closure_end.end_week'):
            node = int(o['closure_end.chokepoint'][j])
            ends[node] = max(ends.get(node, t), int(o['closure_end.end_week'][j]))
        for j in self.live(o, 'pending_prohibitions.effective_week'):
            key = (int(o['pending_prohibitions.edge'][j]), int(o['pending_prohibitions.k'][j]))
            pending[key] = min(pending.get(key, self.T + 1), int(o['pending_prohibitions.effective_week'][j]))
        lots = np.asarray(o.get('queue_lots.qty', []))
        lotmask = np.asarray(o.get('queue_lots.qty.observed', np.ones(lots.shape)), dtype=bool)
        cohorts, queue = [], {}
        for j, entry in enumerate(self.l.get('lot_keys', [])):
            if j >= len(lots):
                break
            node, k, lane, ed = entry
            node, k, ed = int(node), int(k), int(ed)
            lane = -1 if lane is None else int(lane)
            row = np.maximum(0.0, np.nan_to_num(np.asarray(lots[j], dtype=float), nan=0.0, posinf=0.0, neginf=0.0))
            if lotmask.shape == lots.shape:
                row = np.where(lotmask[j], row, 0.0)
            poolkey = (node, self.pool[k])
            queue[poolkey] = queue.get(poolkey, 0.0) + float(np.sum(row))
            for cohort in np.flatnonzero(row > 1e-8):
                cohorts.append((int(cohort), j, node, k, lane, ed, float(row[cohort])))

        def cp_capacity(node, k):
            j = self.cp[node]
            caps = ktb if self.pool[k] == 'tb' else kct
            op = float(opened[j])
            if op < 0.05:
                if node not in ends:
                    return np.inf, 0.0
                return float(max(t, ends[node] + 1)), max(0.0, float(caps[j]))
            return float(t), max(0.0, float(caps[j]) * op)

        def arrival(path, k, start, check=False):
            when = float(start)
            for ed in path:
                node = int(self.tail[ed])
                if node in self.cp:
                    drain_start, cap = cp_capacity(node, k)
                    if cap <= 0 or not np.isfinite(drain_start):
                        return np.inf
                    when = max(when, drain_start)
                    wait = queue.get((node, self.pool[k]), 0.0) / cap
                    when += min(12.0, max(0.0, wait - 0.75 * max(0.0, when - drain_start)))
                if check and (prohibited[ed, k] or pending.get((ed, k), self.T + 1) <= when):
                    return np.inf
                when += max(1.0, float(tau[ed]))
            return when

        def freight(path, k):
            return sum(float(c[ed]) + max(0.0, float(tariff[ed, k])) * self.v[k] for ed in path)

        events, total = {}, stock.copy()
        def add(key, q, when):
            if q > 0:
                events.setdefault(key, []).append((float(when), float(q)))
                total[key] = total.get(key, 0.0) + float(q)

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
                when = arrival(path[path.index(ed) + 1:], k, max(t, when), True)
            key = (dest, k)
            add(key, float(pq[j]), when)
            if pa[j] <= t and dest == int(self.head[ed]):
                available[key] = available.get(key, 0.0) + float(pq[j])

        pool_used, edge_used, pool_cursor = {}, {}, {}
        override_qty = np.zeros(len(self.overrides))
        release_mode = np.zeros(len(self.release_pairs), dtype=np.int64)
        override_mask = np.asarray(o.get('override_mask', np.ones(len(self.overrides))), dtype=bool)
        pair_cohorts, removed = {}, {}
        for item in cohorts:
            pair_cohorts.setdefault((item[2], item[3]), []).append(item)
        for pair, entries in pair_cohorts.items():
            if pair not in self.release_index:
                continue
            node, k = pair
            start, cap = cp_capacity(node, k)
            if start > t or cap <= 1e-8 or not np.isfinite(start):
                continue
            destinations, all_blocked = set(), True
            for cohort, row, cp, good, lane, ed, qty in entries:
                path = self.lanes[lane] if 0 <= lane < len(self.lanes) else [ed]
                path = path[path.index(ed):] if ed in path else [ed]
                destinations.add(int(self.head[path[-1]]))
                if np.isfinite(arrival(path, k, t, True)):
                    all_blocked = False
            if not all_blocked or len(destinations) != 1:
                continue
            dest = next(iter(destinations))
            candidates = []
            for i, (cp, good, ed, lane, path) in enumerate(self.overrides):
                if cp != node or good != k or not override_mask[i] or u[ed] <= 1e-8:
                    continue
                if int(self.head[path[-1]]) != dest or any(prohibited[x, k] for x in path):
                    continue
                when = arrival(path[1:], k, t + max(1.0, float(tau[ed])), True)
                last = self.T if (dest, k) in self.di else self.T - self.downstream.get(dest, 2.0)
                if not np.isfinite(when) or when > last or pending.get((ed, k), self.T + 1) <= t:
                    continue
                cost = freight(path, k)
                if (dest, k) in self.di and not self.carry.get((dest, k), True) and cost >= self.penalty[(dest, k)]:
                    continue
                candidates.append((cost, when, i, ed))
            quantity = sum(x[-1] for x in entries)
            poolkey = (node, self.pool[k])
            share = cap * quantity / max(quantity, queue.get(poolkey, quantity))
            room = min(share, cap - pool_used.get((poolkey, t), 0.0))
            left = quantity
            for cost, when, i, ed in sorted(candidates):
                q = min(left, room, float(u[ed]) - edge_used.get((ed, t), 0.0))
                if q <= 1e-8:
                    continue
                override_qty[i] += q
                release_mode[self.release_index[pair]] = 1
                pool_used[(poolkey, t)] = pool_used.get((poolkey, t), 0.0) + q
                edge_used[(ed, t)] = edge_used.get((ed, t), 0.0) + q
                add((dest, k), q, when)
                left -= q
                room -= q
                if min(room, left) <= 1e-8:
                    break
            amount = quantity - left
            for item in sorted(entries):
                q = min(amount, item[-1])
                removed[(item[0], item[1])] = q
                amount -= q
                if amount <= 1e-8:
                    break

        for cohort, rowindex, node, k, lane, ed, qty in sorted(cohorts):
            qty -= removed.get((cohort, rowindex), 0.0)
            if qty <= 1e-8:
                continue
            path = self.lanes[lane] if 0 <= lane < len(self.lanes) else [ed]
            dest = int(self.head[path[-1]])
            suffix = path[path.index(ed) + 1:] if ed in path else []
            key = (dest, k)
            start, cap = cp_capacity(node, k)
            if not np.isfinite(start) or cap <= 1e-8 or prohibited[ed, k]:
                add(key, qty, np.inf)
                continue
            poolkey = (node, self.pool[k])
            week = max(int(np.ceil(start)), pool_cursor.get(poolkey, t))
            ri = self.release_index.get((node, k))
            if ri is not None and release_mode[ri] == 1:
                week = max(week, t + 1)
            edgecap = float(u[ed])
            if start > t and edgecap <= 1e-8:
                edgecap = float(self.u0[ed])
            if edgecap <= 1e-8:
                add(key, qty, np.inf)
                continue
            while qty > 1e-8 and week <= self.T:
                if pending.get((ed, k), self.T + 1) <= week:
                    break
                pkey, ekey = (poolkey, week), (ed, week)
                room = min(cap - pool_used.get(pkey, 0.0), edgecap - edge_used.get(ekey, 0.0))
                if room <= 1e-8:
                    if cap - pool_used.get(pkey, 0.0) <= 1e-8:
                        pool_cursor[poolkey] = week + 1
                    week += 1
                    continue
                q = min(qty, room)
                pool_used[pkey] = pool_used.get(pkey, 0.0) + q
                edge_used[ekey] = edge_used.get(ekey, 0.0) + q
                add(key, q, arrival(suffix, k, week + max(1.0, float(tau[ed])), True))
                qty -= q
                if cap - pool_used[pkey] <= 1e-8:
                    pool_cursor[poolkey] = week + 1
                if qty > 1e-8:
                    week += 1
            if qty > 1e-8:
                add(key, qty, np.inf)

        wq = np.asarray(o.get('wip.qty', []))
        for j in self.live(o, 'wip.qty'):
            key = (int(o['wip.node'][j]), int(o['wip.k'][j]))
            when = float(o['wip.out_week'][j])
            add(key, float(wq[j]), when)
            if when <= t:
                available[key] = available.get(key, 0.0) + float(wq[j])
        if self.previous is not None:
            incoming, outgoing = {}, {}
            executed = np.maximum(0.0, np.asarray(o.get('last_week.clip.executed', np.zeros(self.n))))
            for i, (origin, dest, k, lane, path) in enumerate(self.routes):
                incoming[(dest, k)] = incoming.get((dest, k), 0.0) + float(executed[i])
                outgoing[(origin, k)] = outgoing.get((origin, k), 0.0) + float(executed[i])
            for key in self.rate:
                if key in self.di:
                    continue
                measurement = max(0.0, self.previous.get(key, 0.0) + incoming.get(key, 0.0) - total.get(key, 0.0), outgoing.get(key, 0.0))
                ceiling = sum(self.u0[self.routes[i][4][0]] for i in self.groups[key])
                measurement = min(measurement, max(ceiling, self.base[key]))
                weight = 0.35 if measurement > self.rate[key] else 0.10
                self.rate[key] = (1.0 - weight) * self.rate[key] + weight * measurement
        self.previous = total.copy()
        ratios = [float(np.mean(forecast[j])) / self.initial_demand[key] for key, j in self.di.items() if 'chip' in str(self.names[key[1]])]
        scale = float(np.clip(np.mean(ratios), 0.7, 1.5)) if ratios else 1.0

        def consumption(key, horizon, rate):
            h = max(0.0, float(horizon))
            if key not in self.di:
                return h * rate
            f = forecast[self.di[key]]
            if not len(f):
                return h * rate
            n = min(int(h), len(f))
            result = float(np.sum(f[:n]))
            if h > n:
                result += (h - n) * (float(f[n]) if n < len(f) else float(np.mean(f)))
            return result

        def lost_before(key, eta, rate):
            if key not in self.di or self.carry.get(key, True):
                return 0.0
            position = stock.get(key, 0.0)
            ev = sorted(events.get(key, []))
            ptr, lost = 0, 0.0
            for week in range(t, min(self.T + 1, int(np.ceil(eta)))):
                while ptr < len(ev) and ev[ptr][0] <= week:
                    position += ev[ptr][1]
                    ptr += 1
                d = consumption(key, week - t + 1, rate) - consumption(key, week - t, rate)
                lost += max(0.0, d - position)
                position = max(0.0, position - d)
            return lost

        def deficit(key, horizon, rate, debt, eta):
            correction = lost_before(key, eta, rate)
            pos = stock.get(key, 0.0) + sum(q for w, q in events.get(key, []) if w <= t + horizon)
            need = max(0.0, consumption(key, horizon, rate) + debt - correction - pos)
            first = max(t, int(np.ceil(eta)))
            stop = min(self.T, int(np.floor(t + horizon - 1.0)))
            for week in range(first, stop + 1):
                pos = stock.get(key, 0.0) + sum(q for w, q in events.get(key, []) if w <= week)
                target = consumption(key, week - t + 1, rate) + debt - correction
                need = max(need, target - pos)
            return max(0.0, need)

        def bridge(key, eta, cheap_eta, rate, debt):
            stop = min(self.T + 1, int(np.ceil(cheap_eta)))
            first = max(t, int(np.ceil(eta)))
            if first >= stop:
                return 0.0
            lost = lost_before(key, eta, rate)
            need = 0.0
            for week in range(first, stop):
                pos = stock.get(key, 0.0) + sum(q for w, q in events.get(key, []) if w <= week)
                need = max(need, consumption(key, week - t + 1, rate) + debt - lost - pos)
            return max(0.0, need)

        requests = []
        for key, ids in self.groups.items():
            dest, k = key
            if key in self.di:
                j = self.di[key]
                rate = max(0.0, float(np.mean(forecast[j])))
                debt = max(0.0, float(backlog[j])) if self.carry.get(key, True) else 0.0
                priority = self.penalty.get(key, self.v[k])
                last, buffer = float(self.T), 2.0
            else:
                floor = 0.60 * self.base[key]
                if 'chip' in str(self.names[k]) or 'wafer' in str(self.names[k]):
                    floor *= scale
                rate = max(self.rate[key], floor)
                debt, priority = 0.0, max(10.0, self.v[k])
                last = self.T - self.downstream.get(dest, 2.0)
                buffer = max(2.75, self.initial_safety.get(key, 0.0))
            if rate <= 1e-8:
                continue
            options, threat_records, survivors = [], [], {}
            for i in ids:
                origin, _, _, lane, path = self.routes[i]
                if not mask[i] or u[path[0]] <= 0 or any(prohibited[ed, k] for ed in path):
                    continue
                cost = freight(path, k)
                nominal_lead = sum(max(1.0, float(tau[ed])) for ed in path)
                effective = min(pending.get((ed, k), self.T + 1) for ed in path)
                if t < effective <= min(self.T, t + nominal_lead + 6.0):
                    threat_records.append((cost, effective, nominal_lead))
                else:
                    survivors[path[0]] = max(survivors.get(path[0], 0.0), float(u[path[0]]))
                source = (origin, k)
                inventory = available.get(source, 0.0)
                if self.types[origin] == 'source' and source not in available:
                    inventory = u[path[0]]
                if inventory <= 1e-8:
                    continue
                eta = arrival(path, k, t, True)
                if not np.isfinite(eta) or eta > last:
                    continue
                lead = max(1.0, eta - t)
                if key in self.di and not self.carry.get(key, True) and cost >= priority:
                    continue
                projected = stock.get(key, 0.0) - debt
                ev = sorted(events.get(key, []))
                ptr, shortage = 0, 0.0
                for week in range(t, int(np.ceil(eta))):
                    while ptr < len(ev) and ev[ptr][0] <= week:
                        projected += ev[ptr][1]
                        ptr += 1
                    projected -= consumption(key, week - t + 1, rate) - consumption(key, week - t, rate)
                    if projected < 0:
                        shortage += min(rate, -projected)
                        projected = 0.0
                score = cost + 0.45 * priority * min(5.0, shortage / rate)
                options.append((score, i, lead, eta, cost))
            if not options:
                continue
            options.sort()
            cheap = min(options, key=lambda x: (x[4], x[2]))
            if threat_records:
                buffer += 2.0
                economical_threat = any(cost <= cheap[4] * 1.15 + 1.0 for cost, effective, lead in threat_records)
                if economical_threat:
                    shortfall = max(0.0, 1.0 - sum(survivors.values()) / max(rate, 1e-8))
                    buffer += 3.0 * shortfall
            horizon = min(cheap[2] + buffer, max(0.0, last - t + 1))
            earliest = min(x[3] for x in options)
            need = deficit(key, horizon, rate, debt, earliest)
            if need > 1e-8:
                requests.append((priority * (1.0 + min(5.0, need / rate)), key, rate, debt, buffer, last, options, cheap))
        requests.sort(key=lambda x: x[0], reverse=True)
        flows, remaining = np.zeros(self.n), u.copy()
        for i, q in enumerate(override_qty):
            if q > 0:
                ed = self.overrides[i][2]
                remaining[ed] = max(0.0, remaining[ed] - q)
        for importance, key, rate, debt, buffer, last, options, cheap in requests:
            horizon = min(cheap[2] + buffer, max(0.0, last - t + 1))
            for score, i, lead, eta, cost in options:
                origin, dest, k, lane, path = self.routes[i]
                source = (origin, k)
                inventory = available.get(source, 0.0)
                if self.types[origin] == 'source' and source not in available:
                    inventory = remaining[path[0]]
                need = deficit(key, horizon, rate, debt, eta)
                local_need = need
                if cost > cheap[4] * 1.10 + 1.0 and lead < cheap[2]:
                    used_sources, used_edges, cheap_capacity = {}, {}, 0.0
                    for alternative in sorted(options, key=lambda x: (x[4], x[2])):
                        if alternative[4] > cheap[4] * 1.10 + 1.0 or alternative[3] > cheap[3] + 1e-8:
                            continue
                        ao, ad, ak, al, ap = self.routes[alternative[1]]
                        akey, ae = (ao, ak), ap[0]
                        inv = available.get(akey, 0.0)
                        if self.types[ao] == 'source' and akey not in available:
                            inv = remaining[ae]
                        q = max(0.0, min(inv - used_sources.get(akey, 0.0), remaining[ae] - used_edges.get(ae, 0.0)))
                        used_sources[akey] = used_sources.get(akey, 0.0) + q
                        used_edges[ae] = used_edges.get(ae, 0.0) + q
                        cheap_capacity += q
                    if cheap_capacity > 0:
                        emergency = bridge(key, eta, cheap[3], rate, debt)
                        premium = cost - cheap[4]
                        if key in self.di:
                            saved_weeks = 1.0 if not self.carry.get(key, True) else max(1.0, cheap[3] - eta)
                            if premium > 0.95 * saved_weeks * self.penalty.get(key, self.v[k]):
                                emergency = 0.0
                        local_need = min(local_need, max(emergency, need - cheap_capacity))
                q = min(local_need, inventory, remaining[path[0]])
                if q <= 1e-8:
                    continue
                flows[i] += q
                remaining[path[0]] -= q
                available[source] = max(0.0, inventory - q)
                events.setdefault(key, []).append((eta, q))
                if deficit(key, horizon, rate, debt, cheap[3]) <= 1e-8:
                    break
        flows = np.maximum(0.0, np.nan_to_num(flows, nan=0.0, posinf=0.0, neginf=0.0))
        override_qty = np.maximum(0.0, np.nan_to_num(override_qty, nan=0.0, posinf=0.0, neginf=0.0))
        return {'flows': flows, 'override_qty': override_qty, 'release_mode': release_mode}
