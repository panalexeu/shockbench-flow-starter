# -2.1085680129591076
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
        self.u0 = np.asarray([0.0 if x is None else float(x) for x in edges['u0']], dtype=float)
        self.c0 = np.asarray([0.0 if x is None else float(x) for x in edges['c0']], dtype=float)
        self.tau0 = np.asarray([1.0 if x is None else float(x) for x in edges['tau0']], dtype=float)
        self.values = np.asarray(s['commodities']['v'], dtype=float)
        self.pools = s['commodities'].get('pool', ['tb'] * len(self.values))
        self.stock_slots = [tuple(map(int, x)) for x in l['stock_slots']]
        self.supply_slots = [tuple(map(int, x)) for x in l['supply_slots']]
        self.demands = [tuple(map(int, x)) for x in l['demands']]
        self.demand_index = {p: i for i, p in enumerate(self.demands)}
        self.cp_index = {int(n): i for i, n in enumerate(l['chokepoints'])}
        self.pi = np.asarray(s['sinks'].get('pi', np.ones(len(self.demands))), dtype=float)
        self.nslots = len(s['action_slots']['edge'])
        self.routes = []

        slots = s['action_slots']
        grouped = {}
        for i, (edge, k, lane) in enumerate(zip(slots['edge'], slots['k'], slots['lane'])):
            edge, k = int(edge), int(k)
            lane = -1 if lane is None else int(lane)
            route_edges = [edge] if lane < 0 else [int(e) for e in s['lanes']['edges'][lane]]
            cps = [] if lane < 0 else [int(cp) for cp in s['lanes']['chokepoints'][lane]]
            if not route_edges:
                continue
            src = int(self.tail[route_edges[0]])
            dst = int(self.head[route_edges[-1]])
            self.routes.append({
                'slot': i, 'k': k, 'lane': lane, 'edges': route_edges,
                'cps': cps, 'src': src, 'dst': dst,
            })
            cap = min((self.u0[e] for e in route_edges), default=0.0)
            alt = edges['alt_of'][edge] if lane < 0 else s['lanes']['alt_of'][lane]
            grouped.setdefault((src, k, dst), []).append((alt, cap))

        # Approximate transfer-node throughput without counting marked alternate
        # routes twice.
        incoming, outgoing = {}, {}
        for (src, k, dst), choices in grouped.items():
            primary = [cap for alt, cap in choices if alt is None]
            cap = sum(primary) if primary else max((x[1] for x in choices), default=0.0)
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
        ncp = len(self.cp_index)

        u = self.field(o, 'graph_now.u', self.u0).astype(float)
        c = self.field(o, 'graph_now.c', self.c0).astype(float)
        tau = self.field(o, 'graph_now.tau', self.tau0).astype(float)
        tariff = self.field(o, 'graph_now.tariff', np.zeros((ne, nk))).astype(float)
        prohibited = self.field(o, 'graph_now.prohibited', np.zeros((ne, nk))).astype(bool)
        opened = self.field(o, 'graph_now.open', np.ones(ncp)).astype(float)
        ktb = self.field(o, 'graph_now.kappa.tb', np.full(ncp, np.inf)).astype(float)
        kct = self.field(o, 'graph_now.kappa.ct', np.full(ncp, np.inf)).astype(float)
        mask = np.asarray(o.get('action_mask', np.ones(self.nslots)), dtype=bool).reshape(-1)

        stock = self.field(o, 'stock.qty', np.zeros(len(self.stock_slots))).astype(float)
        stock_now = {}
        available = {}
        for i, p in enumerate(self.stock_slots):
            q = max(0.0, float(stock[i])) if i < stock.size else 0.0
            stock_now[p] = q
            available[p] = q

        supply = self.field(o, 'graph_now.supply.avail', np.zeros(len(self.supply_slots))).astype(float)
        for i, p in enumerate(self.supply_slots):
            q = max(0.0, float(supply[i])) if i < supply.size else 0.0
            stock_now[p] = stock_now.get(p, 0.0) + q
            available[p] = available.get(p, 0.0) + q

        # Future arrivals are tracked separately from dispatchable inventory.
        projected = {}
        def add_projection(p, arrival, qty):
            qty = max(0.0, float(qty))
            if qty > 0:
                projected.setdefault(p, []).append((int(arrival), qty))

        pqty = np.asarray(o.get('pipeline.qty', []), dtype=float).reshape(-1)
        plive = np.asarray(o.get('pipeline.qty.observed', np.ones(pqty.size)), dtype=bool).reshape(-1)
        pedge = np.asarray(o.get('pipeline.edge', np.zeros(pqty.size)), dtype=int).reshape(-1)
        pk = np.asarray(o.get('pipeline.k', np.zeros(pqty.size)), dtype=int).reshape(-1)
        plane = np.asarray(o.get('pipeline.lane', np.full(pqty.size, -1)), dtype=int).reshape(-1)
        plane_obs = np.asarray(o.get('pipeline.lane.observed', np.ones(pqty.size)), dtype=bool).reshape(-1)
        parr = np.asarray(o.get('pipeline.arrival_week', np.full(pqty.size, week)), dtype=int).reshape(-1)
        lanes = self.s['lanes']['edges']
        npipe = min(pqty.size, pedge.size, pk.size, parr.size)
        for j in range(npipe):
            if j >= plive.size or not plive[j] or pqty[j] <= 0 or not 0 <= pedge[j] < ne:
                continue
            lane = int(plane[j]) if j < plane.size and j < plane_obs.size and plane_obs[j] else -1
            dst = int(self.head[pedge[j]])
            arrival = int(parr[j])
            if 0 <= lane < len(lanes):
                les = [int(e) for e in lanes[lane]]
                dst = int(self.head[les[-1]])
                if int(pedge[j]) in les:
                    pos = les.index(int(pedge[j]))
                    arrival += int(np.ceil(sum(max(1.0, float(tau[e])) for e in les[pos + 1:])))
            add_projection((dst, int(pk[j])), max(week, arrival), pqty[j])

        wqty = np.asarray(o.get('wip.qty', []), dtype=float).reshape(-1)
        wlive = np.asarray(o.get('wip.qty.observed', np.ones(wqty.size)), dtype=bool).reshape(-1)
        wnode = np.asarray(o.get('wip.node', np.zeros(wqty.size)), dtype=int).reshape(-1)
        wk = np.asarray(o.get('wip.k', np.zeros(wqty.size)), dtype=int).reshape(-1)
        wout = np.asarray(o.get('wip.out_week', np.full(wqty.size, week)), dtype=int).reshape(-1)
        for j in range(min(wqty.size, wnode.size, wk.size, wout.size)):
            if j < wlive.size and wlive[j] and wqty[j] > 0:
                add_projection((int(wnode[j]), int(wk[j])), max(week, int(wout[j])), wqty[j])

        # Lots already waiting at a chokepoint should count as future supply.
        if 'queue_lots.qty' in o:
            lots = np.asarray(o['queue_lots.qty'], dtype=float)
            live = np.asarray(o.get('queue_lots.qty.observed', np.ones(lots.shape)), dtype=bool)
            for row, key in enumerate(self.l.get('lot_keys', [])[:lots.shape[0]]):
                if len(key) < 4:
                    continue
                _, commodity, lane, next_edge = key
                commodity, next_edge = int(commodity), int(next_edge)
                if not 0 <= next_edge < ne:
                    continue
                lane = -1 if lane is None else int(lane)
                if 0 <= lane < len(lanes):
                    les = [int(e) for e in lanes[lane]]
                    dst = int(self.head[les[-1]])
                    rem = sum(max(1.0, float(tau[e])) for e in les[les.index(next_edge):]) if next_edge in les else max(1.0, float(tau[next_edge]))
                else:
                    dst = int(self.head[next_edge])
                    rem = max(1.0, float(tau[next_edge]))
                for col in range(lots.shape[1]):
                    if row < live.shape[0] and col < live.shape[1] and live[row, col] and lots[row, col] > 0:
                        add_projection((dst, commodity), max(week, col + 1) + int(np.ceil(rem)), lots[row, col])

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

        def forecast_at(di, h):
            if forecast.ndim != 2 or di >= forecast.shape[0] or forecast.shape[1] == 0:
                return 0.0
            row = forecast[di]
            if h < row.size:
                return max(0.0, float(row[h]))
            tail = row[-min(3, row.size):]
            return max(0.0, float(np.mean(tail)))

        def sink_need(di, lead):
            node, k = self.demands[di]
            # Cover the visible forecast and extend the reorder window when
            # transit takes longer than the forecast horizon.
            remain = max(0, self.T - week + 1)
            horizon = min(remain, max(1, int(np.ceil(lead)) + 8))
            demand_total = max(0.0, float(backlog[di])) if di < backlog.size else 0.0
            for h in range(horizon):
                demand_total += forecast_at(di, h)
            cutoff = week + horizon - 1
            inventory = stock_now.get((node, k), 0.0)
            inventory += sum(q for t, q in projected.get((node, k), ()) if t <= cutoff)
            return max(0.0, demand_total - inventory), cutoff

        # Chokepoint throughput is shared across all action slots using its pool.
        cp_left = {}
        for cp, ci in self.cp_index.items():
            for pool, caps in (('tb', ktb), ('ct', kct)):
                cap = float(caps[ci]) if ci < len(caps) else np.inf
                cp_left[(cp, pool)] = max(0.0, cap * max(0.0, float(opened[ci])))
        edge_left = np.maximum(0.0, u.copy())

        # Build the set of currently feasible arcs and shortest downstream
        # cost-to-sink estimates for each commodity.
        feasible = []
        by_k = {}
        for r in self.routes:
            i, k, es, cps = r['slot'], r['k'], r['edges'], r['cps']
            if i >= mask.size or not mask[i] or not es:
                continue
            if any(e < 0 or e >= ne for e in es):
                continue
            if any(k < prohibited.shape[1] and prohibited[e, k] for e in es):
                continue
            lead = sum(max(1.0, float(tau[e])) for e in es)
            if week + int(np.ceil(lead)) > self.T:
                continue
            if any(cp in self.cp_index and opened[self.cp_index[cp]] <= 0.001 for cp in cps):
                continue
            risk, elapsed = False, 0.0
            for e in es:
                if pending.get((e, k), self.T + 1) <= week + elapsed:
                    risk = True
                    break
                elapsed += max(1.0, float(tau[e]))
            if risk:
                continue
            cap = min((max(0.0, float(u[e])) for e in es), default=0.0)
            for cp in cps:
                if cp in self.cp_index:
                    cap = min(cap, cp_left.get((cp, self.pools[k]), 0.0))
            if cap <= 0:
                continue
            cost = sum(float(c[e]) + float(tariff[e, k]) * float(self.values[k]) for e in es)
            generalized = cost + 0.002 * float(self.values[k]) * lead
            item = dict(r)
            item.update({'lead': lead, 'cap': cap, 'cost': cost, 'gen': generalized})
            feasible.append(item)
            by_k.setdefault(k, []).append(item)

        # Reverse Dijkstra gives the cheapest remaining route from every node
        # to each sink; these distances make transfer routes value-aware.
        downstream = {}
        for di, (sink, k) in enumerate(self.demands):
            arcs = by_k.get(k, [])
            reverse = {}
            for r in arcs:
                reverse.setdefault(r['dst'], []).append((r['src'], r['gen'], r['lead']))
            dist, times = {sink: 0.0}, {sink: 0.0}
            heap = [(0.0, sink)]
            while heap:
                d, node = heapq.heappop(heap)
                if d > dist.get(node, float('inf')) + 1e-12:
                    continue
                for prev, weight, lead in reverse.get(node, ()):
                    nd = d + weight
                    if nd + 1e-12 < dist.get(prev, float('inf')):
                        dist[prev] = nd
                        times[prev] = times[node] + lead
                        heapq.heappush(heap, (nd, prev))
            downstream[(di, k)] = (dist, times)

        candidates = []
        for r in feasible:
            k, dst = r['k'], r['dst']
            if (dst, k) in self.demand_index:
                di = self.demand_index[(dst, k)]
                need, _ = sink_need(di, r['lead'])
                penalty = max(1.0, float(self.pi[di]) if di < self.pi.size else 1.0)
                score = r['gen'] / penalty
                if need > 0:
                    candidates.append((score, r['lead'], r['slot'], r, di))
                continue

            rate = max(0.0, float(self.node_rate.get((dst, k), 0.0)))
            if rate <= 0:
                continue
            best_score = float('inf')
            best_time = 0.0
            for di, (sink, dk) in enumerate(self.demands):
                if dk != k:
                    continue
                dist, times = downstream.get((di, k), ({}, {}))
                if dst not in dist:
                    continue
                penalty = max(1.0, float(self.pi[di]) if di < self.pi.size else 1.0)
                value = (r['gen'] + dist[dst]) / penalty
                if value < best_score:
                    best_score, best_time = value, times.get(dst, 0.0)
            if not np.isfinite(best_score):
                continue
            duration = min(4.0, max(2.0, 1.0 + 0.5 * best_time))
            target = rate * duration
            have = stock_now.get((dst, k), 0.0)
            have += sum(q for t, q in projected.get((dst, k), ()) if t <= week + int(np.ceil(duration)))
            if target > have:
                candidates.append((best_score, r['lead'], r['slot'], r, None))

        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        flows = np.zeros(self.nslots, dtype=float)
        for _, _, _, r, di in candidates:
            k, src, dst = r['k'], r['src'], r['dst']
            if di is not None:
                need, _ = sink_need(di, r['lead'])
            else:
                rate = max(0.0, float(self.node_rate.get((dst, k), 0.0)))
                best_time = 0.0
                best_score = float('inf')
                for sj, (sink, sk) in enumerate(self.demands):
                    if sk != k:
                        continue
                    dist, times = downstream.get((sj, k), ({}, {}))
                    if dst in dist:
                        penalty = max(1.0, float(self.pi[sj]) if sj < self.pi.size else 1.0)
                        value = (r['gen'] + dist[dst]) / penalty
                        if value < best_score:
                            best_score, best_time = value, times.get(dst, 0.0)
                duration = min(4.0, max(2.0, 1.0 + 0.5 * best_time))
                target = rate * duration
                cutoff = week + int(np.ceil(duration))
                have = stock_now.get((dst, k), 0.0)
                have += sum(q for t, q in projected.get((dst, k), ()) if t <= cutoff)
                need = max(0.0, target - have)

            have_source = max(0.0, available.get((src, k), 0.0))
            q = min(float(r['cap']), need, have_source)
            for e in r['edges']:
                q = min(q, max(0.0, float(edge_left[e])))
            for cp in r['cps']:
                if cp in self.cp_index:
                    q = min(q, cp_left.get((cp, self.pools[k]), 0.0))
            if not np.isfinite(q) or q <= 0:
                continue

            flows[r['slot']] += q
            available[(src, k)] = max(0.0, have_source - q)
            stock_now[(src, k)] = max(0.0, stock_now.get((src, k), 0.0) - q)
            for e in r['edges']:
                edge_left[e] = max(0.0, edge_left[e] - q)
            for cp in r['cps']:
                if cp in self.cp_index:
                    key = (cp, self.pools[k])
                    cp_left[key] = max(0.0, cp_left.get(key, 0.0) - q)
            add_projection((dst, k), week + int(np.ceil(r['lead'])), q)

        return {'flows': flows}
