# 0.4959127949022634
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
        self.pool = s['commodities'].get('pool', ['tb'] * len(self.values))
        self.stocks = [tuple(map(int, p)) for p in l['stock_slots']]
        self.supplies = [tuple(map(int, p)) for p in l['supply_slots']]
        self.demands = [tuple(map(int, p)) for p in l['demands']]
        self.di = {p: i for i, p in enumerate(self.demands)}
        self.cp_index = {int(n): i for i, n in enumerate(l['chokepoints'])}
        self.pi = np.asarray(s['sinks'].get('pi', np.ones(len(self.demands))), dtype=float)
        self.max_pi = max(1.0, float(np.max(self.pi)) if self.pi.size else 1.0)

        self.routes = []
        slots = s['action_slots']
        grouped = {}
        for i, (edge, k, lane) in enumerate(zip(slots['edge'], slots['k'], slots['lane'])):
            edge, k = int(edge), int(k)
            lane = -1 if lane is None else int(lane)
            es = [edge] if lane < 0 else [int(x) for x in s['lanes']['edges'][lane]]
            cps = [] if lane < 0 else [int(x) for x in s['lanes']['chokepoints'][lane]]
            if not es:
                continue
            src, dst = int(self.tail[es[0]]), int(self.head[es[-1]])
            alt = edges['alt_of'][edge] if lane < 0 else s['lanes']['alt_of'][lane]
            r = {'i': i, 'k': k, 'edges': es, 'cps': cps, 'src': src, 'dst': dst, 'alt': alt}
            self.routes.append(r)
            cap = min((self.u0[e] for e in es), default=0.0)
            grouped.setdefault((src, k, dst), []).append((alt, cap))

        incoming, outgoing = {}, {}
        reverse = {}
        for (src, k, dst), choices in grouped.items():
            base = [cap for alt, cap in choices if alt is None]
            cap = sum(base) if base else max((x[1] for x in choices), default=0.0)
            incoming[(dst, k)] = incoming.get((dst, k), 0.0) + cap
            outgoing[(src, k)] = outgoing.get((src, k), 0.0) + cap
        self.node_rate = {}
        for p in set(incoming) | set(outgoing):
            a, b = incoming.get(p), outgoing.get(p)
            self.node_rate[p] = min(a, b) if a is not None and b is not None else max(a or 0.0, b or 0.0)
        for r in self.routes:
            reverse.setdefault((r['dst'], r['k']), set()).add(r['src'])
        self.node_priority = {}
        for di, (sink, k) in enumerate(self.demands):
            penalty = max(1.0, float(self.pi[di]) if di < self.pi.size else 1.0)
            queue, seen = [(int(sink), 0)], {int(sink): 0}
            pos = 0
            while pos < len(queue):
                node, dist = queue[pos]
                pos += 1
                key = (node, int(k))
                self.node_priority[key] = max(self.node_priority.get(key, 0.0), penalty / (1.0 + 0.12 * dist))
                for prev in reverse.get(key, ()):
                    if prev not in seen:
                        seen[prev] = dist + 1
                        queue.append((prev, dist + 1))

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
        nslot = len(self.s['action_slots']['edge'])
        u = self.field(o, 'graph_now.u', self.u0).astype(float)
        c = self.field(o, 'graph_now.c', self.c0).astype(float)
        tau = self.field(o, 'graph_now.tau', self.tau0).astype(float)
        tariff = self.field(o, 'graph_now.tariff', np.zeros((ne, nk))).astype(float)
        opened = self.field(o, 'graph_now.open', np.ones(len(self.cp_index))).astype(float)
        ktb = self.field(o, 'graph_now.kappa.tb', np.full(len(self.cp_index), np.inf)).astype(float)
        kct = self.field(o, 'graph_now.kappa.ct', np.full(len(self.cp_index), np.inf)).astype(float)
        mask = np.asarray(o.get('action_mask', np.ones(nslot)), dtype=bool).reshape(-1)

        stock = self.field(o, 'stock.qty', np.zeros(len(self.stocks))).astype(float)
        available = {}
        stock_at = {}
        for i, p in enumerate(self.stocks):
            q = max(0.0, float(stock[i])) if i < stock.size else 0.0
            available[p] = q
            stock_at[p] = q
        supply = self.field(o, 'graph_now.supply.avail', np.zeros(len(self.supplies))).astype(float)
        for i, p in enumerate(self.supplies):
            if i < supply.size:
                available[p] = available.get(p, 0.0) + max(0.0, float(supply[i]))

        projected = {}
        def add_projection(p, arrival, q):
            if q > 0:
                projected.setdefault(p, []).append((int(arrival), float(q)))

        pqty = np.asarray(o.get('pipeline.qty', [])).reshape(-1)
        plive = np.asarray(o.get('pipeline.qty.observed', np.ones(pqty.size)), dtype=bool).reshape(-1)
        pedge = np.asarray(o.get('pipeline.edge', np.zeros(pqty.size)), dtype=int).reshape(-1)
        pk = np.asarray(o.get('pipeline.k', np.zeros(pqty.size)), dtype=int).reshape(-1)
        plane = np.asarray(o.get('pipeline.lane', np.full(pqty.size, -1)), dtype=int).reshape(-1)
        plane_obs = np.asarray(o.get('pipeline.lane.observed', np.ones(pqty.size)), dtype=bool).reshape(-1)
        parr = np.asarray(o.get('pipeline.arrival_week', np.full(pqty.size, week)), dtype=int).reshape(-1)
        lanes = self.s['lanes']['edges']
        for j in range(min(pqty.size, pedge.size, pk.size, parr.size)):
            if j >= plive.size or not plive[j] or pqty[j] <= 0 or not 0 <= pedge[j] < ne:
                continue
            lane = int(plane[j]) if j < plane.size and j < plane_obs.size and plane_obs[j] else -1
            dst = int(self.head[pedge[j]])
            arrival = int(parr[j])
            if 0 <= lane < len(lanes):
                les = [int(x) for x in lanes[lane]]
                dst = int(self.head[les[-1]])
                if pedge[j] in les:
                    pos = les.index(int(pedge[j]))
                    arrival += int(np.ceil(sum(max(1.0, float(tau[e])) for e in les[pos + 1:])))
            add_projection((dst, int(pk[j])), max(week, arrival), pqty[j])

        wqty = np.asarray(o.get('wip.qty', [])).reshape(-1)
        wlive = np.asarray(o.get('wip.qty.observed', np.ones(wqty.size)), dtype=bool).reshape(-1)
        wnode = np.asarray(o.get('wip.node', np.zeros(wqty.size)), dtype=int).reshape(-1)
        wk = np.asarray(o.get('wip.k', np.zeros(wqty.size)), dtype=int).reshape(-1)
        wout = np.asarray(o.get('wip.out_week', np.full(wqty.size, week)), dtype=int).reshape(-1)
        for j in range(min(wqty.size, wnode.size, wk.size, wout.size)):
            if j < wlive.size and wlive[j] and wqty[j] > 0:
                add_projection((int(wnode[j]), int(wk[j])), max(week, int(wout[j])), wqty[j])

        if 'queue_lots.qty' in o:
            lots = np.asarray(o['queue_lots.qty'], dtype=float)
            live = np.asarray(o.get('queue_lots.qty.observed', np.ones(lots.shape)), dtype=bool)
            for row, key in enumerate(self.l.get('lot_keys', [])[:lots.shape[0]]):
                if len(key) < 4:
                    continue
                _, k, lane, next_edge = key
                next_edge = int(next_edge)
                if not 0 <= next_edge < ne:
                    continue
                lane = -1 if lane is None else int(lane)
                if 0 <= lane < len(lanes):
                    les = [int(x) for x in lanes[lane]]
                    dst = int(self.head[les[-1]])
                    rem = sum(max(1.0, float(tau[e])) for e in les[les.index(next_edge):]) if next_edge in les else max(1.0, float(tau[next_edge]))
                else:
                    dst, rem = int(self.head[next_edge]), max(1.0, float(tau[next_edge]))
                for col in range(lots.shape[1]):
                    if row < live.shape[0] and col < live.shape[1] and live[row, col] and lots[row, col] > 0:
                        add_projection((dst, int(k)), max(week, col + 1) + int(np.ceil(rem)), lots[row, col])

        forecast = np.asarray(o.get('demand_forecast.qty', np.zeros((len(self.demands), 1))), dtype=float)
        backlog = np.asarray(o.get('backlog.qty', np.zeros(len(self.demands))), dtype=float).reshape(-1)

        def demand_through(di, cutoff):
            if di >= backlog.size:
                base = 0.0
            else:
                base = max(0.0, float(backlog[di]))
            if forecast.ndim != 2 or di >= forecast.shape[0] or forecast.shape[1] == 0:
                return base
            h = max(0, int(cutoff) - week + 1)
            row = forecast[di]
            n = min(h, row.size)
            total = base + float(np.sum(row[:n]))
            if h > n:
                tail = row[-min(3, row.size):]
                total += (h - n) * max(0.0, float(np.mean(tail)))
            return total

        def need_sink(di, cutoff):
            p = self.demands[di]
            have = stock_at.get(p, 0.0)
            have += sum(q for arrival, q in projected.get(p, ()) if arrival <= cutoff)
            return max(0.0, demand_through(di, cutoff) - have)

        cp_left = {}
        for cp, ci in self.cp_index.items():
            for pool, arr in (('tb', ktb), ('ct', kct)):
                cap = float(arr[ci]) if ci < len(arr) else np.inf
                cp_left[(cp, pool)] = max(0.0, cap * max(0.0, float(opened[ci])))
        edge_left = np.maximum(0.0, u.copy())

        # Keep only legal, usable action arcs. Their cost and lead time also
        # provide a weighted shortest-path metric to each demand sink.
        arcs_by_k = {}
        arc_info = {}
        for r in self.routes:
            i, k, es, cps = r['i'], r['k'], r['edges'], r['cps']
            if i >= mask.size or not mask[i] or any(e < 0 or e >= ne for e in es):
                continue
            cap = min((max(0.0, float(u[e])) for e in es), default=0.0)
            lead = sum(max(1.0, float(tau[e])) for e in es)
            cost = sum(float(c[e]) + float(tariff[e, k]) * float(self.values[k]) for e in es)
            for cp in cps:
                ci = self.cp_index.get(cp)
                if ci is not None:
                    cap = min(cap, cp_left.get((cp, self.pool[k]), 0.0))
            if cap <= 0 or lead <= 0:
                continue
            r['cap'] = cap
            r['lead'] = lead
            r['cost'] = cost
            arcs_by_k.setdefault(k, {}).setdefault(r['src'], []).append(r)
            arc_info[i] = r

        flows = np.zeros(nslot, dtype=float)
        candidates = []
        # Build least-cost paths from each currently stocked node to each sink.
        source_nodes = {(node, k) for node, k in available if available[(node, k)] > 0}
        for src, k in source_nodes:
            adj = arcs_by_k.get(k, {})
            if not adj:
                continue
            dist = {src: 0.0}
            prev = {}
            heap = [(0.0, src)]
            while heap:
                d, node = heapq.heappop(heap)
                if d != dist.get(node):
                    continue
                for r in adj.get(node, ()):
                    nd = d + r['cost'] + 0.002 * float(self.values[k]) * r['lead']
                    if nd < dist.get(r['dst'], float('inf')):
                        dist[r['dst']] = nd
                        prev[r['dst']] = (node, r)
                        heapq.heappush(heap, (nd, r['dst']))
            for di, (sink, dk) in enumerate(self.demands):
                if dk != k or sink not in dist or sink == src:
                    continue
                path = []
                node = sink
                seen = set()
                while node != src and node in prev and node not in seen:
                    seen.add(node)
                    parent, r = prev[node]
                    path.append(r)
                    node = parent
                if node != src or not path:
                    continue
                path.reverse()
                lead = sum(r['lead'] for r in path)
                arrival = week + int(np.ceil(lead))
                if arrival > self.T:
                    continue
                bottleneck = min(r['cap'] for r in path)
                penalty = max(1.0, float(self.pi[di]) if di < self.pi.size else 1.0)
                score = dist[sink] / penalty
                candidates.append((score, lead, 'sink', path, di, bottleneck))

        # Maintain modest buffers at transfer/material nodes, including nodes
        # serving production chains whose final commodity differs from inputs.
        for r in self.routes:
            if r['dst'] in self.di or r['i'] not in arc_info:
                continue
            priority = max(1.0, self.node_priority.get((r['dst'], r['k']), self.max_pi * 0.20))
            score = (r['cost'] + 0.002 * float(self.values[r['k']]) * r['lead']) / priority
            candidates.append((score, r['lead'], 'buffer', [r], r['dst'], r['cap']))

        candidates.sort(key=lambda x: (x[0], x[1], x[3][0]['i']))
        for _, lead, kind, path, target, planned_cap in candidates:
            first = path[0]
            k, src = first['k'], first['src']
            if kind == 'sink':
                di = target
                arrival = week + int(np.ceil(lead))
                qty_needed = need_sink(di, arrival)
            else:
                p = (int(target), k)
                duration = min(max(2.0, lead + 2.0), max(1.0, self.T - week + 1))
                target_qty = max(0.0, float(self.node_rate.get(p, 0.0))) * duration
                cutoff = week + int(np.ceil(duration))
                have_at_dest = stock_at.get(p, 0.0) + sum(q for t, q in projected.get(p, ()) if t <= cutoff)
                qty_needed = max(0.0, target_qty - have_at_dest)
                arrival = week + int(np.ceil(lead))
            have = max(0.0, available.get((src, k), 0.0))
            q = min(float(planned_cap), qty_needed, have)
            for e in first['edges']:
                q = min(q, max(0.0, float(edge_left[e])))
            for cp in first['cps']:
                if cp in self.cp_index:
                    q = min(q, cp_left.get((cp, self.pool[k]), 0.0))
            if q <= 0 or not np.isfinite(q):
                continue
            flows[first['i']] += q
            available[(src, k)] = max(0.0, have - q)
            for e in first['edges']:
                edge_left[e] = max(0.0, edge_left[e] - q)
            for cp in first['cps']:
                if cp in self.cp_index:
                    key = (cp, self.pool[k])
                    cp_left[key] = max(0.0, cp_left.get(key, 0.0) - q)
            if kind == 'sink':
                add_projection(self.demands[target], arrival, q)
            else:
                add_projection((int(target), k), arrival, q)

        return {'flows': flows}
