# -2.108726119187684
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
        self.pool = s['commodities'].get('pool', ['tb'] * len(self.values))
        self.stocks = [tuple(map(int, p)) for p in l['stock_slots']]
        self.supplies = [tuple(map(int, p)) for p in l['supply_slots']]
        self.demands = [tuple(map(int, p)) for p in l['demands']]
        self.demand_index = {p: i for i, p in enumerate(self.demands)}
        self.cp_index = {int(n): i for i, n in enumerate(l['chokepoints'])}
        self.pi = np.asarray(s['sinks'].get('pi', np.ones(len(self.demands))), dtype=float)
        self.max_pi = max(1.0, float(np.max(self.pi)) if self.pi.size else 1.0)

        self.routes = []
        slots = s['action_slots']
        grouped = {}
        reverse = {}
        for i, (edge, commodity, lane) in enumerate(
                zip(slots['edge'], slots['k'], slots['lane'])):
            edge, k = int(edge), int(commodity)
            lane = -1 if lane is None else int(lane)
            es = [edge] if lane < 0 else [int(x) for x in s['lanes']['edges'][lane]]
            cps = [] if lane < 0 else [int(x) for x in s['lanes']['chokepoints'][lane]]
            if not es:
                continue
            src, dst = int(self.tail[es[0]]), int(self.head[es[-1]])
            alt = edges['alt_of'][edge] if lane < 0 else s['lanes']['alt_of'][lane]
            r = {'i': i, 'k': k, 'edges': es, 'cps': cps,
                 'src': src, 'dst': dst, 'alt': alt}
            self.routes.append(r)
            cap = min((self.u0[e] for e in es), default=0.0)
            grouped.setdefault((src, k, dst), []).append((alt, cap))
            reverse.setdefault((dst, k), set()).add(src)

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

        self.node_priority = {}
        for di, (sink, k) in enumerate(self.demands):
            penalty = max(1.0, float(self.pi[di]) if di < self.pi.size else 1.0)
            queue, seen = [(int(sink), 0)], {int(sink)}
            pos = 0
            while pos < len(queue):
                node, distance = queue[pos]
                pos += 1
                key = (node, int(k))
                self.node_priority[key] = max(
                    self.node_priority.get(key, 0.0), penalty / (1.0 + 0.15 * distance))
                for prev in reverse.get(key, ()):
                    if prev not in seen:
                        seen.add(prev)
                        queue.append((prev, distance + 1))

    @staticmethod
    def _field(obs, name, fallback):
        fb = np.asarray(fallback)
        if name not in obs:
            return fb.copy()
        x = np.asarray(obs[name])
        mask = obs.get(name + '.observed')
        if mask is not None and np.shape(mask) == x.shape:
            return np.where(mask, x, fb)
        return x.copy()

    def act(self, obs):
        week = int(np.asarray(obs['week']).reshape(-1)[0])
        nedge, nk = len(self.tail), len(self.values)
        nslot = len(self.s['action_slots']['edge'])
        ncp = len(self.cp_index)
        u = self._field(obs, 'graph_now.u', self.u0).astype(float)
        c = self._field(obs, 'graph_now.c', self.c0).astype(float)
        tau = self._field(obs, 'graph_now.tau', self.tau0).astype(float)
        tariff = self._field(obs, 'graph_now.tariff', np.zeros((nedge, nk))).astype(float)
        prohibited = self._field(obs, 'graph_now.prohibited', np.zeros((nedge, nk))).astype(bool)
        opened = self._field(obs, 'graph_now.open', np.ones(ncp)).astype(float)
        ktb = self._field(obs, 'graph_now.kappa.tb', np.full(ncp, np.inf)).astype(float)
        kct = self._field(obs, 'graph_now.kappa.ct', np.full(ncp, np.inf)).astype(float)
        mask = np.asarray(obs.get('action_mask', np.ones(nslot)), dtype=bool).reshape(-1)

        pending = {}
        pe = np.asarray(obs.get('pending_prohibitions.edge', []), dtype=int).reshape(-1)
        pk = np.asarray(obs.get('pending_prohibitions.k', np.zeros(pe.size)), dtype=int).reshape(-1)
        pw = np.asarray(obs.get('pending_prohibitions.effective_week',
                                 np.full(pe.size, self.T + 1)), dtype=int).reshape(-1)
        pm = np.asarray(obs.get('pending_prohibitions.edge.observed',
                                np.ones(pe.size)), dtype=bool).reshape(-1)
        for j in range(min(pe.size, pk.size, pw.size, pm.size)):
            if pm[j]:
                key = (int(pe[j]), int(pk[j]))
                pending[key] = min(pending.get(key, self.T + 1), int(pw[j]))

        available = {}
        projected = {}

        def add_projection(pair, arrival, qty):
            if qty > 0:
                projected.setdefault(pair, []).append((int(arrival), float(qty)))

        stock = self._field(obs, 'stock.qty', np.zeros(len(self.stocks))).astype(float)
        for i, pair in enumerate(self.stocks):
            q = max(0.0, float(stock[i])) if i < stock.size else 0.0
            available[pair] = q
            add_projection(pair, week, q)

        supply = self._field(obs, 'graph_now.supply.avail', np.zeros(len(self.supplies))).astype(float)
        for i, pair in enumerate(self.supplies):
            q = max(0.0, float(supply[i])) if i < supply.size else 0.0
            available[pair] = available.get(pair, 0.0) + q
            add_projection(pair, week, q)

        lanes = self.s['lanes']['edges']
        pqty = np.asarray(obs.get('pipeline.qty', [])).reshape(-1)
        plive = np.asarray(obs.get('pipeline.qty.observed', np.ones(pqty.size)), dtype=bool).reshape(-1)
        pedge = np.asarray(obs.get('pipeline.edge', np.zeros(pqty.size)), dtype=int).reshape(-1)
        pkp = np.asarray(obs.get('pipeline.k', np.zeros(pqty.size)), dtype=int).reshape(-1)
        plane = np.asarray(obs.get('pipeline.lane', np.full(pqty.size, -1)), dtype=int).reshape(-1)
        plane_obs = np.asarray(obs.get('pipeline.lane.observed', np.ones(pqty.size)), dtype=bool).reshape(-1)
        parr = np.asarray(obs.get('pipeline.arrival_week', np.full(pqty.size, week)), dtype=int).reshape(-1)
        for j in range(min(pqty.size, pedge.size, pkp.size, parr.size)):
            if j >= plive.size or not plive[j] or pqty[j] <= 0 or not 0 <= pedge[j] < nedge:
                continue
            lane = int(plane[j]) if j < plane.size and j < plane_obs.size and plane_obs[j] else -1
            dst = int(self.head[pedge[j]])
            arrival = int(parr[j])
            if 0 <= lane < len(lanes):
                es = [int(e) for e in lanes[lane]]
                dst = int(self.head[es[-1]])
                if int(pedge[j]) in es:
                    pos = es.index(int(pedge[j]))
                    arrival += int(np.ceil(sum(max(1.0, float(tau[e])) for e in es[pos + 1:])))
            add_projection((dst, int(pkp[j])), max(week, arrival), pqty[j])

        wqty = np.asarray(obs.get('wip.qty', [])).reshape(-1)
        wlive = np.asarray(obs.get('wip.qty.observed', np.ones(wqty.size)), dtype=bool).reshape(-1)
        wnode = np.asarray(obs.get('wip.node', np.zeros(wqty.size)), dtype=int).reshape(-1)
        wk = np.asarray(obs.get('wip.k', np.zeros(wqty.size)), dtype=int).reshape(-1)
        wout = np.asarray(obs.get('wip.out_week', np.full(wqty.size, week)), dtype=int).reshape(-1)
        for j in range(min(wqty.size, wnode.size, wk.size, wout.size)):
            if j < wlive.size and wlive[j] and wqty[j] > 0:
                add_projection((int(wnode[j]), int(wk[j])), max(week, int(wout[j])), wqty[j])

        if 'queue_lots.qty' in obs:
            lots = np.asarray(obs['queue_lots.qty'], dtype=float)
            live = np.asarray(obs.get('queue_lots.qty.observed', np.ones(lots.shape)), dtype=bool)
            for row, key in enumerate(self.l.get('lot_keys', [])[:lots.shape[0]]):
                if len(key) < 4:
                    continue
                _, commodity, lane, next_edge = key
                commodity, next_edge = int(commodity), int(next_edge)
                if not 0 <= next_edge < nedge:
                    continue
                lane = -1 if lane is None else int(lane)
                if 0 <= lane < len(lanes):
                    es = [int(e) for e in lanes[lane]]
                    dst = int(self.head[es[-1]])
                    rem = (sum(max(1.0, float(tau[e])) for e in es[es.index(next_edge):])
                           if next_edge in es else max(1.0, float(tau[next_edge])))
                else:
                    dst, rem = int(self.head[next_edge]), max(1.0, float(tau[next_edge]))
                for col in range(lots.shape[1]):
                    if (row < live.shape[0] and col < live.shape[1] and live[row, col]
                            and lots[row, col] > 0):
                        arrival = max(week, col + 1) + int(np.ceil(rem))
                        add_projection((dst, commodity), arrival, lots[row, col])

        forecast = np.asarray(obs.get('demand_forecast.qty',
                                      np.zeros((len(self.demands), 1))), dtype=float)
        backlog = np.asarray(obs.get('backlog.qty', np.zeros(len(self.demands))), dtype=float).reshape(-1)

        def projected_qty(pair, cutoff):
            return sum(q for arrival, q in projected.get(pair, ()) if arrival <= cutoff)

        def demand_through(di, cutoff):
            base = max(0.0, float(backlog[di])) if di < backlog.size else 0.0
            if forecast.ndim != 2 or di >= forecast.shape[0] or forecast.shape[1] == 0:
                return base
            count = max(0, int(cutoff) - week + 1)
            row = forecast[di]
            n = min(count, row.size)
            total = base + float(np.sum(row[:n]))
            if count > n:
                tail = row[-min(3, row.size):]
                total += (count - n) * max(0.0, float(np.mean(tail)))
            return total

        def sink_need(di, arrival):
            cutoff = min(self.T, int(arrival) + 1)
            return max(0.0, demand_through(di, cutoff) -
                       projected_qty(self.demands[di], cutoff))

        cp_left = {}
        for cp, ci in self.cp_index.items():
            for pool_name, arr in (('tb', ktb), ('ct', kct)):
                cap = float(arr[ci]) if ci < arr.size else np.inf
                cp_left[(cp, pool_name)] = max(0.0, cap * max(0.0, float(opened[ci])))
        edge_left = np.maximum(0.0, u.copy())

        # Build legal current-week arcs. A route's edge capacities and chokepoint
        # throughput constrain its first dispatch; pending prohibitions are
        # checked at the estimated time each edge would be entered.
        adj = {}
        for r in self.routes:
            i, k, es = r['i'], r['k'], r['edges']
            if i >= mask.size or not mask[i] or any(e < 0 or e >= nedge for e in es):
                continue
            if any(k >= prohibited.shape[1] or prohibited[e, k] for e in es):
                continue
            cap = min((max(0.0, float(u[e])) for e in es), default=0.0)
            lead = sum(max(1.0, float(tau[e])) for e in es)
            cost = 0.0
            for e in es:
                if k < tariff.shape[1]:
                    cost += float(c[e]) + float(tariff[e, k]) * float(self.values[k])
                else:
                    cost += float(c[e])
            for cp in r['cps']:
                ci = self.cp_index.get(cp)
                if ci is not None:
                    cap = min(cap, cp_left.get((cp, self.pool[k]), 0.0))
            if cap <= 0 or lead <= 0:
                continue
            r['cap'], r['lead'], r['cost'] = cap, lead, cost
            adj.setdefault((k, r['src']), []).append(r)

        def shortest_path(src, k, sink):
            start = (int(k), int(src))
            dist = {int(src): 0.0}
            elapsed = {int(src): 0.0}
            prev = {}
            heap = [(0.0, int(src))]
            while heap:
                d, node = heapq.heappop(heap)
                if d > dist.get(node, float('inf')) + 1e-9:
                    continue
                if node == sink:
                    break
                for r in adj.get((int(k), node), ()):
                    offset = 0.0
                    blocked = False
                    for e in r['edges']:
                        if pending.get((e, int(k)), self.T + 1) <= week + elapsed[node] + offset:
                            blocked = True
                            break
                        offset += max(1.0, float(tau[e]))
                    if blocked:
                        continue
                    nd = d + max(0.0, r['cost']) + 0.002 * float(self.values[k]) * r['lead']
                    if nd < dist.get(r['dst'], float('inf')) - 1e-9:
                        dist[r['dst']] = nd
                        elapsed[r['dst']] = elapsed[node] + r['lead']
                        prev[r['dst']] = (node, r)
                        heapq.heappush(heap, (nd, r['dst']))
            if sink not in dist or sink == src:
                return None
            path, node = [], int(sink)
            seen = set()
            while node != src and node in prev and node not in seen:
                seen.add(node)
                parent, r = prev[node]
                path.append(r)
                node = parent
            if node != src:
                return None
            path.reverse()
            return dist[sink], elapsed[sink], path

        candidates = []
        for (src, k), qty in list(available.items()):
            if qty <= 0:
                continue
            for di, (sink, dk) in enumerate(self.demands):
                if int(dk) != int(k) or int(sink) == int(src):
                    continue
                result = shortest_path(int(src), int(k), int(sink))
                if result is None:
                    continue
                path_cost, lead, path = result
                arrival = week + int(np.ceil(lead))
                if arrival > self.T:
                    continue
                cap = min((r['cap'] for r in path), default=0.0)
                if cap <= 0:
                    continue
                penalty = max(1.0, float(self.pi[di]) if di < self.pi.size else 1.0)
                candidates.append((path_cost / penalty, lead, 'sink', path, di, cap))

        # A modest buffer objective keeps useful transfer nodes supplied when
        # there is no immediate sink shipment competing for the same stock.
        for r in self.routes:
            if r['i'] not in {x['i'] for x in self.routes if 'cap' in x}:
                continue
            p = (r['dst'], r['k'])
            if p in self.demand_index:
                continue
            priority = self.node_priority.get(p, 0.0)
            if priority <= 0:
                continue
            score = (max(0.0, r['cost']) + 0.002 * float(self.values[r['k']]) * r['lead']) / max(1.0, priority)
            candidates.append((score, r['lead'], 'buffer', [r], p, r['cap']))

        candidates.sort(key=lambda x: (x[0], x[1], x[3][0]['i']))
        flows = np.zeros(nslot, dtype=float)
        for _, lead, kind, path, target, planned_cap in candidates:
            first = path[0]
            k, src = first['k'], first['src']
            if kind == 'sink':
                di = int(target)
                destination = self.demands[di]
                arrival = week + int(np.ceil(lead))
                need = sink_need(di, arrival)
            else:
                destination = target
                duration = min(max(2.0, lead + 2.0), max(1.0, self.T - week + 1))
                cutoff = min(self.T, week + int(np.ceil(duration)))
                target_qty = max(0.0, float(self.node_rate.get(destination, 0.0))) * duration
                need = max(0.0, target_qty - projected_qty(destination, cutoff))
                arrival = week + int(np.ceil(lead))

            have = max(0.0, available.get((src, k), 0.0))
            q = min(float(planned_cap), need, have)
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

            # Track the physical first-leg arrival, and reserve the sink arrival
            # for a multi-leg plan so another source does not duplicate it.
            first_arrival = week + int(np.ceil(first['lead']))
            add_projection((first['dst'], k), first_arrival, q)
            if kind == 'sink' and destination != (first['dst'], k):
                add_projection(destination, arrival, q)

        return {'flows': flows}
