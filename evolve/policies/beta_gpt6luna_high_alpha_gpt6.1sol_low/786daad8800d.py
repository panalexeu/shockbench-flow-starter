# -1.9514088415663071
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
        self.node_types = s.get('nodes', {}).get('type', [])
        self.stock_slots = [tuple(map(int, x)) for x in l.get('stock_slots', [])]
        self.supply_slots = [tuple(map(int, x)) for x in l.get('supply_slots', [])]
        self.demands = [tuple(map(int, x)) for x in l.get('demands', [])]
        self.demand_index = {p: i for i, p in enumerate(self.demands)}
        self.cp_index = {int(n): i for i, n in enumerate(l.get('chokepoints', []))}
        self.nslots = len(s['action_slots'].get('edge', []))
        self.nedges = len(self.tail)
        self.ncommodities = len(self.values)

        sink_pi = {}
        sinks = s.get('sinks', {})
        for n, k, pi in zip(sinks.get('node', []), sinks.get('k', []), sinks.get('pi', [])):
            sink_pi[(int(n), int(k))] = max(1.0, float(pi))
        self.pi = np.asarray([sink_pi.get(p, 1.0) for p in self.demands])
        self.max_pi = max(1.0, float(np.max(self.pi)) if self.pi.size else 1.0)

        slots = s['action_slots']
        self.routes = []
        self.nominal_groups = {}
        for i, (edge, k, lane) in enumerate(zip(slots['edge'], slots['k'], slots['lane'])):
            edge, k = int(edge), int(k)
            lane = -1 if lane is None else int(lane)
            if lane < 0:
                route_edges = [edge]
                cps = []
                alt = edges.get('alt_of', [None] * self.nedges)[edge]
            else:
                route_edges = [int(x) for x in s['lanes']['edges'][lane]]
                cps = [int(x) for x in s['lanes']['chokepoints'][lane]]
                alt = s['lanes'].get('alt_of', [None] * len(s['lanes']['edges']))[lane]
            if not route_edges:
                continue
            src = int(self.tail[route_edges[0]])
            dst = int(self.head[route_edges[-1]])
            r = {'slot': i, 'k': k, 'edges': route_edges, 'cps': cps,
                 'src': src, 'dst': dst, 'lane': lane}
            self.routes.append(r)
            cap = min((self.u0[e] for e in route_edges), default=0.0)
            self.nominal_groups.setdefault((src, k, dst), []).append((alt, cap))

        # Estimate normal intermediate-node throughput without double-counting
        # explicitly marked alternate routes.
        incoming, outgoing = {}, {}
        for (src, k, dst), choices in self.nominal_groups.items():
            base = [cap for alt, cap in choices if alt is None]
            rate = sum(base) if base else max((cap for _, cap in choices), default=0.0)
            incoming[(dst, k)] = incoming.get((dst, k), 0.0) + rate
            outgoing[(src, k)] = outgoing.get((src, k), 0.0) + rate
        self.node_rate = {}
        for pair in set(incoming) | set(outgoing):
            a, b = incoming.get(pair), outgoing.get(pair)
            self.node_rate[pair] = min(a, b) if a is not None and b is not None else max(a or 0.0, b or 0.0)

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
        s, l = self.s, self.l
        u = self.field(o, 'graph_now.u', self.u0).astype(float).reshape(-1)
        c = self.field(o, 'graph_now.c', self.c0).astype(float).reshape(-1)
        tau = np.maximum(1.0, self.field(o, 'graph_now.tau', self.tau0).astype(float).reshape(-1))
        tariff = self.field(o, 'graph_now.tariff', np.zeros((self.nedges, self.ncommodities))).astype(float)
        prohibited = self.field(o, 'graph_now.prohibited', np.zeros((self.nedges, self.ncommodities))).astype(bool)
        opened = self.field(o, 'graph_now.open', np.ones(len(self.cp_index))).astype(float).reshape(-1)
        action_mask = np.asarray(o.get('action_mask', np.ones(self.nslots)), dtype=bool).reshape(-1)
        stock_qty = self.field(o, 'stock.qty', np.zeros(len(self.stock_slots))).astype(float).reshape(-1)
        supply_qty = self.field(o, 'graph_now.supply.avail', np.zeros(len(self.supply_slots))).astype(float).reshape(-1)
        forecast = np.asarray(o.get('demand_forecast.qty', np.zeros((len(self.demands), 1))), dtype=float)
        backlog = np.asarray(o.get('backlog.qty', np.zeros(len(self.demands))), dtype=float).reshape(-1)

        # Dispatchable stock is separate from future projected inventory.
        available = {}
        projected = {}
        for i, pair in enumerate(self.stock_slots):
            q = max(0.0, float(stock_qty[i])) if i < stock_qty.size else 0.0
            available[pair] = q
            if q > 0:
                projected.setdefault(pair, []).append((week, q))
        for i, pair in enumerate(self.supply_slots):
            q = max(0.0, float(supply_qty[i])) if i < supply_qty.size else 0.0
            available[pair] = available.get(pair, 0.0) + q

        lanes = s.get('lanes', {}).get('edges', [])
        pq = np.asarray(o.get('pipeline.qty', [])).reshape(-1)
        p_live = np.asarray(o.get('pipeline.qty.observed', np.ones(pq.size)), dtype=bool).reshape(-1)
        p_edge = np.asarray(o.get('pipeline.edge', np.zeros(pq.size)), dtype=int).reshape(-1)
        p_k = np.asarray(o.get('pipeline.k', np.zeros(pq.size)), dtype=int).reshape(-1)
        p_lane = np.asarray(o.get('pipeline.lane', np.full(pq.size, -1)), dtype=int).reshape(-1)
        p_lane_live = np.asarray(o.get('pipeline.lane.observed', np.ones(pq.size)), dtype=bool).reshape(-1)
        p_arrival = np.asarray(o.get('pipeline.arrival_week', np.full(pq.size, week)), dtype=int).reshape(-1)
        for j in range(min(pq.size, p_edge.size, p_k.size, p_arrival.size)):
            if j >= p_live.size or not p_live[j] or pq[j] <= 0 or not 0 <= p_edge[j] < self.nedges:
                continue
            lane = int(p_lane[j]) if j < p_lane.size and j < p_lane_live.size and p_lane_live[j] else -1
            dst = int(self.head[p_edge[j]])
            arrival = int(p_arrival[j])
            if 0 <= lane < len(lanes) and lanes[lane]:
                les = [int(x) for x in lanes[lane]]
                dst = int(self.head[les[-1]])
                if int(p_edge[j]) in les:
                    pos = les.index(int(p_edge[j]))
                    arrival += int(np.ceil(sum(tau[e] for e in les[pos + 1:])))
            projected.setdefault((dst, int(p_k[j])), []).append((max(week, arrival), float(pq[j])))

        wq = np.asarray(o.get('wip.qty', [])).reshape(-1)
        w_live = np.asarray(o.get('wip.qty.observed', np.ones(wq.size)), dtype=bool).reshape(-1)
        w_node = np.asarray(o.get('wip.node', np.zeros(wq.size)), dtype=int).reshape(-1)
        w_k = np.asarray(o.get('wip.k', np.zeros(wq.size)), dtype=int).reshape(-1)
        w_out = np.asarray(o.get('wip.out_week', np.full(wq.size, week)), dtype=int).reshape(-1)
        for j in range(min(wq.size, w_node.size, w_k.size, w_out.size)):
            if j < w_live.size and w_live[j] and wq[j] > 0:
                projected.setdefault((int(w_node[j]), int(w_k[j])), []).append((max(week, int(w_out[j])), float(wq[j])))

        # Estimate arrivals of cargo already waiting at chokepoints.
        lots = np.asarray(o.get('queue_lots.qty', np.zeros((0, 0))), dtype=float)
        lot_live = np.asarray(o.get('queue_lots.qty.observed', np.ones(lots.shape)), dtype=bool)
        closure_end = {}
        ce = np.asarray(o.get('closure_end.chokepoint', []), dtype=int).reshape(-1)
        ce_live = np.asarray(o.get('closure_end.chokepoint.observed', np.zeros(ce.size)), dtype=bool).reshape(-1)
        ew = np.asarray(o.get('closure_end.end_week', np.zeros(ce.size)), dtype=int).reshape(-1)
        ew_live = np.asarray(o.get('closure_end.end_week.observed', np.zeros(ew.size)), dtype=bool).reshape(-1)
        for j in range(min(ce.size, ew.size, ce_live.size, ew_live.size)):
            if ce_live[j] and ew_live[j]:
                closure_end[int(ce[j])] = int(ew[j]) + 1
        for row, key in enumerate(l.get('lot_keys', [])[:lots.shape[0]]):
            if len(key) < 4 or row >= lot_live.shape[0]:
                continue
            cp, k, lane, next_edge = key
            cp, k, next_edge = int(cp), int(k), int(next_edge)
            if not 0 <= next_edge < self.nedges:
                continue
            lane = -1 if lane is None else int(lane)
            les = [int(x) for x in lanes[lane]] if 0 <= lane < len(lanes) else [next_edge]
            dst = int(self.head[les[-1]])
            rem = sum(tau[e] for e in les[les.index(next_edge):]) if next_edge in les else tau[next_edge]
            for col in range(min(lots.shape[1], lot_live.shape[1])):
                q = float(lots[row, col]) if lot_live[row, col] else 0.0
                if q <= 0:
                    continue
                release = max(week, col + 1)
                ci = self.cp_index.get(cp)
                if ci is not None and ci < opened.size and opened[ci] <= 0.001:
                    release = max(release, closure_end.get(cp, week + 3))
                arrival = release + int(np.ceil(rem))
                projected.setdefault((dst, k), []).append((arrival, q))

        pending = {}
        pe = np.asarray(o.get('pending_prohibitions.edge', [])).reshape(-1).astype(int)
        pk = np.asarray(o.get('pending_prohibitions.k', np.zeros(pe.size))).reshape(-1).astype(int)
        pw = np.asarray(o.get('pending_prohibitions.effective_week', np.full(pe.size, self.T + 1))).reshape(-1).astype(int)
        pm = np.asarray(o.get('pending_prohibitions.edge.observed', np.ones(pe.size)), dtype=bool).reshape(-1)
        for j in range(min(pe.size, pk.size, pw.size)):
            if j < pm.size and pm[j]:
                pair = (int(pe[j]), int(pk[j]))
                pending[pair] = min(pending.get(pair, self.T + 1), int(pw[j]))

        warnings = {}
        warning = self.field(o, 'warning.score', np.zeros(len(l.get('warning_units', [])))).reshape(-1)
        for i, key in enumerate(l.get('warning_units', [])):
            if i < warning.size:
                warnings[tuple(key)] = float(np.clip(warning[i], 0.0, 1.0))

        def projected_by(pair, cutoff):
            return sum(q for arrival, q in projected.get(pair, []) if arrival <= cutoff)

        def demand_target(di, cutoff):
            # Include backlog and forecast demand from this week through cutoff.
            target = max(0.0, float(backlog[di])) if di < backlog.size else 0.0
            n = max(1, int(cutoff) - week + 1)
            if forecast.ndim == 2 and di < forecast.shape[0] and forecast.shape[1] > 0:
                row = forecast[di]
                use = min(n, row.size)
                if use:
                    target += float(np.sum(row[:use]))
                if n > use:
                    tail = row[-min(3, row.size):]
                    target += (n - use) * max(0.0, float(np.mean(tail)))
            return target

        # Route capacities and cost use the observations for this week.
        cp_left = {}
        for pool in ('tb', 'ct'):
            caps = self.field(o, 'graph_now.kappa.' + pool, np.full(len(self.cp_index), np.inf)).astype(float).reshape(-1)
            for cp, ci in self.cp_index.items():
                cap = caps[ci] if ci < caps.size else np.inf
                op = opened[ci] if ci < opened.size else 1.0
                cp_left[(cp, pool)] = max(0.0, float(cap) * max(0.0, float(op)))
        edge_left = np.maximum(0.0, u.copy())
        valid = []
        adjacency = {}
        for r in self.routes:
            i, k, es, cps = r['slot'], r['k'], r['edges'], r['cps']
            if i >= action_mask.size or not action_mask[i] or any(prohibited[e, k] for e in es):
                continue
            lead = float(sum(tau[e] for e in es))
            cap = min((max(0.0, float(u[e])) for e in es), default=0.0)
            if cap <= 0 or week + lead > self.T + 1:
                continue
            crossing_time = 0.0
            blocked = False
            for e in es:
                if pending.get((e, k), self.T + 1) <= week + crossing_time:
                    blocked = True
                    break
                crossing_time += tau[e]
            if blocked:
                continue
            for cp in cps:
                ci = self.cp_index.get(cp)
                if ci is not None:
                    op = opened[ci] if ci < opened.size else 1.0
                    cap = min(cap, cp_left.get((cp, self.pools[k]), 0.0))
                    if op <= 0.001:
                        cap = 0.0
            if cap <= 0:
                continue
            freight = 0.0
            for e in es:
                rate = float(tariff[e, k]) if tariff.ndim == 2 and e < tariff.shape[0] and k < tariff.shape[1] else 0.0
                val = float(self.values[k]) if k < self.values.size else 0.0
                freight += float(c[e]) + rate * val
            risk = max([warnings.get(('chokepoint', cp), 0.0) for cp in cps] or [0.0])
            # A modest risk premium breaks ties without ignoring current freight costs.
            val = float(self.values[k]) if k < self.values.size else 0.0
            cost = max(0.0, freight) * (1.0 + 0.20 * risk) + 0.002 * val * lead
            rr = dict(r, lead=lead, cap=cap, cost=cost)
            valid.append(rr)
            adjacency.setdefault((k, r['src']), []).append(rr)

        # Shortest route plans from each dispatchable source to all reachable
        # nodes, retaining route-level arcs so only the first move is acted on.
        paths = {}
        source_pairs = [p for p, q in available.items() if q > 1e-9]
        for src, k in source_pairs:
            start = (src, k)
            dist = {src: 0.0}
            prev = {}
            heap = [(0.0, src)]
            while heap:
                d, node = heapq.heappop(heap)
                if d > dist.get(node, float('inf')) + 1e-10:
                    continue
                for r in adjacency.get((k, node), []):
                    nd = d + r['cost']
                    if nd < dist.get(r['dst'], float('inf')) - 1e-10:
                        dist[r['dst']] = nd
                        prev[r['dst']] = (node, r)
                        heapq.heappush(heap, (nd, r['dst']))
            for dst, d in dist.items():
                if dst == src:
                    continue
                path = []
                node = dst
                seen = set()
                while node != src and node in prev and node not in seen:
                    seen.add(node)
                    parent, rr = prev[node]
                    path.append(rr)
                    node = parent
                if node != src or not path:
                    continue
                path.reverse()
                phys = sum(rr['lead'] for rr in path)
                effective = phys + max(0, len(path) - 1)
                arrival = week + int(np.ceil(effective))
                if arrival <= self.T:
                    paths[(src, k, dst)] = (path, effective, arrival, d)

        candidates = []
        # Demand-directed plans: use the shortage penalty to prioritize scarce
        # commodities and reserve projected sink demand as plans are dispatched.
        for di, (sink, k) in enumerate(self.demands):
            pi = max(1.0, float(self.pi[di]))
            for src, sk in source_pairs:
                if sk != k or src == sink:
                    continue
                plan = paths.get((src, k, sink))
                if plan is None:
                    continue
                path, effective, arrival, route_cost = plan
                cutoff = min(self.T, arrival + 2)
                target = demand_target(di, cutoff)
                gap = max(0.0, target - projected_by((sink, k), cutoff))
                if gap <= 1e-9:
                    continue
                priority = (route_cost + 0.002 * float(self.values[k]) * effective) / pi
                candidates.append((priority, arrival, 0, di, src, k, sink, path, cutoff))

        # Production inputs may have no same-commodity path to a demand sink.
        # Maintain a small, capacity-scaled buffer at production/material nodes.
        for src, k in source_pairs:
            for dst in range(len(self.node_types)):
                if dst == src or (dst, k) in self.demand_index:
                    continue
                typ = self.node_types[dst]
                if typ not in ('fab', 'osat', 'material'):
                    continue
                plan = paths.get((src, k, dst))
                if plan is None:
                    continue
                path, effective, arrival, route_cost = plan
                cutoff = min(self.T, arrival + 1)
                rate = max(0.0, float(self.node_rate.get((dst, k), 0.0)))
                if rate <= 0:
                    rate = max((rr['cap'] for rr in path), default=0.0)
                duration = min(3.0, max(1.0, effective + 1.0), max(1.0, self.T - week + 1))
                target = rate * duration
                gap = max(0.0, target - projected_by((dst, k), cutoff))
                if gap <= 1e-9:
                    continue
                priority = (route_cost + 0.002 * float(self.values[k]) * effective) / max(1.0, self.max_pi * 0.18)
                candidates.append((priority, arrival, 1, -1, src, k, dst, path, cutoff))

        candidates.sort(key=lambda x: (x[0], x[1], x[2], x[4], x[6]))
        flows = np.zeros(self.nslots, dtype=float)
        for _, arrival, kind, di, src, k, dst, path, cutoff in candidates:
            if not path:
                continue
            first = path[0]
            pair = (dst, k)
            if kind == 0:
                target = demand_target(di, cutoff)
                need = max(0.0, target - projected_by(pair, cutoff))
            else:
                rate = max(0.0, float(self.node_rate.get(pair, 0.0)))
                if rate <= 0:
                    rate = max((rr['cap'] for rr in path), default=0.0)
                duration = min(3.0, max(1.0, first['lead'] + 1.0), max(1.0, self.T - week + 1))
                need = max(0.0, rate * duration - projected_by(pair, cutoff))
            q = min(float(first['cap']), max(0.0, available.get((src, k), 0.0)), need)
            for e in first['edges']:
                q = min(q, max(0.0, float(edge_left[e])))
            for cp in first['cps']:
                q = min(q, cp_left.get((cp, self.pools[k]), 0.0))
            if q <= 1e-9 or not np.isfinite(q):
                continue
            flows[first['slot']] += q
            available[(src, k)] = max(0.0, available.get((src, k), 0.0) - q)
            for e in first['edges']:
                edge_left[e] = max(0.0, edge_left[e] - q)
            for cp in first['cps']:
                key = (cp, self.pools[k])
                cp_left[key] = max(0.0, cp_left.get(key, 0.0) - q)
            # Reserve the eventual sink/buffer arrival so later candidates do
            # not send duplicate quantities against the same projected need.
            projected.setdefault(pair, []).append((arrival, q))

        return {'flows': flows}
