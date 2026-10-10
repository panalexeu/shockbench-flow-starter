# 0.48913787594283203
import heapq
import numpy as np


class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.T = int(config['T'])
        s, l = self.s, self.l
        e = s['edges']
        self.tail = np.asarray(e['tail'], dtype=int)
        self.head = np.asarray(e['head'], dtype=int)
        self.u0 = np.asarray([0.0 if x is None else float(x) for x in e['u0']])
        self.c0 = np.asarray([0.0 if x is None else float(x) for x in e['c0']])
        self.tau0 = np.asarray([1.0 if x is None else float(x) for x in e['tau0']])
        self.values = np.asarray(s['commodities']['v'], dtype=float)
        self.pools = s['commodities'].get('pool', ['tb'] * len(self.values))
        self.stock_slots = [tuple(map(int, p)) for p in l['stock_slots']]
        self.supply_slots = [tuple(map(int, p)) for p in l['supply_slots']]
        self.demands = [tuple(map(int, p)) for p in l['demands']]
        self.demand_index = {p: i for i, p in enumerate(self.demands)}
        self.cp_index = {int(n): i for i, n in enumerate(l['chokepoints'])}
        self.nslots = len(s['action_slots']['edge'])
        self.routes = []
        slots = s['action_slots']
        for i, (edge, k, lane) in enumerate(zip(slots['edge'], slots['k'], slots['lane'])):
            edge, k = int(edge), int(k)
            lane = -1 if lane is None else int(lane)
            if lane < 0:
                es, cps = [edge], []
            else:
                es = [int(x) for x in s['lanes']['edges'][lane]]
                cps = [int(x) for x in s['lanes']['chokepoints'][lane]]
            if not es:
                continue
            self.routes.append({
                'slot': i, 'k': k, 'edges': es, 'cps': cps,
                'src': int(self.tail[es[0]]), 'dst': int(self.head[es[-1]])
            })

        # A rough steady-state throughput estimate supports conservative
        # intermediate buffers when a commodity has no reachable demand sink.
        grouped = {}
        for r in self.routes:
            cap = min((self.u0[x] for x in r['edges']), default=0.0)
            grouped.setdefault((r['src'], r['k'], r['dst']), []).append((cap, r))
        incoming, outgoing = {}, {}
        for (src, k, dst), choices in grouped.items():
            edge_table = s['edges']
            slot = choices[0][1]['slot']
            action_edge = int(slots['edge'][slot])
            lane = slots['lane'][slot]
            alt = edge_table['alt_of'][action_edge] if lane is None else s['lanes']['alt_of'][int(lane)]
            base = [cap for cap, rr in choices
                    if (edge_table['alt_of'][int(slots['edge'][rr['slot']])] if slots['lane'][rr['slot']] is None
                        else s['lanes']['alt_of'][int(slots['lane'][rr['slot']])]) is None]
            cap = sum(base) if base else max((x[0] for x in choices), default=0.0)
            incoming[(dst, k)] = incoming.get((dst, k), 0.0) + cap
            outgoing[(src, k)] = outgoing.get((src, k), 0.0) + cap
        self.node_rate = {}
        for p in set(incoming) | set(outgoing):
            a, b = incoming.get(p), outgoing.get(p)
            self.node_rate[p] = min(a, b) if a is not None and b is not None else max(a or 0.0, b or 0.0)

        sink_table = s.get('sinks', {})
        pi_by_pair = {}
        for n, k, pi in zip(sink_table.get('node', []), sink_table.get('k', []), sink_table.get('pi', [])):
            pi_by_pair[(int(n), int(k))] = max(1.0, float(pi))
        self.pi = np.asarray([pi_by_pair.get(p, 1.0) for p in self.demands], dtype=float)
        self.max_pi = max(1.0, float(np.max(self.pi)) if self.pi.size else 1.0)

        # Reverse reachability gives intermediate nodes a fallback priority.
        reverse = {}
        for r in self.routes:
            reverse.setdefault((r['dst'], r['k']), set()).add(r['src'])
        self.node_priority = {}
        for di, (sink, k) in enumerate(self.demands):
            queue = [(sink, 0)]
            seen = {sink}
            for node, dist in queue:
                p = (node, k)
                self.node_priority[p] = max(self.node_priority.get(p, 0.0), self.pi[di] / (1.0 + 0.12 * dist))
                for prev in reverse.get(p, ()):
                    if prev not in seen:
                        seen.add(prev)
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
        u = self.field(o, 'graph_now.u', self.u0).astype(float)
        c = self.field(o, 'graph_now.c', self.c0).astype(float)
        tau = np.maximum(1.0, self.field(o, 'graph_now.tau', self.tau0).astype(float))
        tariff = self.field(o, 'graph_now.tariff', np.zeros((ne, nk))).astype(float)
        prohibited = self.field(o, 'graph_now.prohibited', np.zeros((ne, nk))).astype(bool)
        opened = self.field(o, 'graph_now.open', np.ones(len(self.cp_index))).astype(float)
        mask = np.asarray(o.get('action_mask', np.ones(self.nslots)), dtype=bool).reshape(-1)
        stock_qty = self.field(o, 'stock.qty', np.zeros(len(self.stock_slots))).reshape(-1)
        supply_qty = self.field(o, 'graph_now.supply.avail', np.zeros(len(self.supply_slots))).reshape(-1)
        forecast = np.asarray(o.get('demand_forecast.qty', np.zeros((len(self.demands), 1))), dtype=float)
        backlog = np.asarray(o.get('backlog.qty', np.zeros(len(self.demands))), dtype=float).reshape(-1)

        available, projected = {}, {}
        for i, p in enumerate(self.stock_slots):
            q = max(0.0, float(stock_qty[i])) if i < stock_qty.size else 0.0
            available[p] = q
            if q > 0:
                projected.setdefault(p, []).append((week, q))
        for i, p in enumerate(self.supply_slots):
            if i < supply_qty.size:
                available[p] = available.get(p, 0.0) + max(0.0, float(supply_qty[i]))

        def add_projection(p, arrival, qty):
            if qty > 0:
                projected.setdefault(p, []).append((int(arrival), float(qty)))

        lanes = self.s['lanes']['edges']
        pq = np.asarray(o.get('pipeline.qty', [])).reshape(-1)
        plive = np.asarray(o.get('pipeline.qty.observed', np.ones(pq.size)), dtype=bool).reshape(-1)
        pe = np.asarray(o.get('pipeline.edge', np.zeros(pq.size)), dtype=int).reshape(-1)
        pk = np.asarray(o.get('pipeline.k', np.zeros(pq.size)), dtype=int).reshape(-1)
        plane = np.asarray(o.get('pipeline.lane', np.full(pq.size, -1)), dtype=int).reshape(-1)
        plane_obs = np.asarray(o.get('pipeline.lane.observed', np.ones(pq.size)), dtype=bool).reshape(-1)
        parr = np.asarray(o.get('pipeline.arrival_week', np.full(pq.size, week)), dtype=int).reshape(-1)
        for j in range(min(pq.size, pe.size, pk.size, parr.size)):
            if j >= plive.size or not plive[j] or pq[j] <= 0 or not 0 <= pe[j] < ne:
                continue
            lane = int(plane[j]) if j < plane.size and j < plane_obs.size and plane_obs[j] else -1
            dst = int(self.head[pe[j]])
            arrival = int(parr[j])
            if 0 <= lane < len(lanes):
                es = [int(x) for x in lanes[lane]]
                dst = int(self.head[es[-1]])
                if int(pe[j]) in es:
                    pos = es.index(int(pe[j]))
                    arrival += int(np.ceil(sum(tau[e] for e in es[pos + 1:])))
            add_projection((dst, int(pk[j])), max(week, arrival), pq[j])

        wq = np.asarray(o.get('wip.qty', [])).reshape(-1)
        wlive = np.asarray(o.get('wip.qty.observed', np.ones(wq.size)), dtype=bool).reshape(-1)
        wn = np.asarray(o.get('wip.node', np.zeros(wq.size)), dtype=int).reshape(-1)
        wk = np.asarray(o.get('wip.k', np.zeros(wq.size)), dtype=int).reshape(-1)
        wo = np.asarray(o.get('wip.out_week', np.full(wq.size, week)), dtype=int).reshape(-1)
        for j in range(min(wq.size, wn.size, wk.size, wo.size)):
            if j < wlive.size and wlive[j] and wq[j] > 0:
                add_projection((int(wn[j]), int(wk[j])), max(week, int(wo[j])), wq[j])

        # Existing lots wait at a chokepoint; estimate their eventual arrival.
        lots = np.asarray(o.get('queue_lots.qty', np.zeros((0, 0))), dtype=float)
        lot_live = np.asarray(o.get('queue_lots.qty.observed', np.ones(lots.shape)), dtype=bool)
        closure_end = {}
        ce = np.asarray(o.get('closure_end.chokepoint', []), dtype=int).reshape(-1)
        cem = np.asarray(o.get('closure_end.chokepoint.observed', np.zeros(ce.size)), dtype=bool).reshape(-1)
        ew = np.asarray(o.get('closure_end.end_week', np.zeros(ce.size)), dtype=int).reshape(-1)
        ewm = np.asarray(o.get('closure_end.end_week.observed', np.zeros(ce.size)), dtype=bool).reshape(-1)
        for j in range(min(ce.size, cem.size, ew.size, ewm.size)):
            if cem[j] and ewm[j]:
                closure_end[int(ce[j])] = int(ew[j]) + 1
        for row, key in enumerate(self.l.get('lot_keys', [])[:lots.shape[0]]):
            if len(key) < 4 or row >= lot_live.shape[0]:
                continue
            cp, k, lane, next_edge = key
            next_edge = int(next_edge)
            if not 0 <= next_edge < ne:
                continue
            lane = -1 if lane is None else int(lane)
            es = [int(x) for x in lanes[lane]] if 0 <= lane < len(lanes) else [next_edge]
            dst = int(self.head[es[-1]])
            rem = sum(tau[e] for e in es[es.index(next_edge):]) if next_edge in es else tau[next_edge]
            q = float(np.sum(np.where(lot_live[row], lots[row], 0.0)))
            if q <= 0:
                continue
            ci = self.cp_index.get(int(cp))
            release = week
            if ci is not None and opened[ci] <= 0.001:
                release = max(release, closure_end.get(int(cp), week + 3))
            add_projection((dst, int(k)), release + int(np.ceil(rem)), q)

        # Earliest announced future prohibition per edge and commodity.
        pending = {}
        p_edge = np.asarray(o.get('pending_prohibitions.edge', []), dtype=int).reshape(-1)
        p_k = np.asarray(o.get('pending_prohibitions.k', np.zeros(p_edge.size)), dtype=int).reshape(-1)
        p_week = np.asarray(o.get('pending_prohibitions.effective_week', np.full(p_edge.size, self.T + 1)), dtype=int).reshape(-1)
        p_live = np.asarray(o.get('pending_prohibitions.edge.observed', np.ones(p_edge.size)), dtype=bool).reshape(-1)
        for j in range(min(p_edge.size, p_k.size, p_week.size)):
            if j < p_live.size and p_live[j]:
                key = (int(p_edge[j]), int(p_k[j]))
                pending[key] = min(pending.get(key, self.T + 1), int(p_week[j]))

        def covered(pair, cutoff):
            return sum(q for arrival, q in projected.get(pair, ()) if arrival <= cutoff)

        def demand_total(di, horizon):
            total = max(0.0, float(backlog[di])) if di < backlog.size else 0.0
            if forecast.ndim != 2 or di >= forecast.shape[0] or forecast.shape[1] == 0:
                return total
            row = forecast[di]
            n = min(max(0, int(horizon)), row.size)
            if n:
                total += float(np.sum(row[:n]))
            if horizon > n and row.size:
                tail = row[-min(3, row.size):]
                total += (horizon - n) * max(0.0, float(np.mean(tail)))
            return total

        # Construct currently usable arcs and shared capacity budgets.
        cp_left = {}
        for pool in ('tb', 'ct'):
            caps = self.field(o, 'graph_now.kappa.' + pool, np.full(len(self.cp_index), np.inf)).astype(float)
            for cp, ci in self.cp_index.items():
                cap = caps[ci] if ci < caps.size else np.inf
                cp_left[(cp, pool)] = max(0.0, float(cap) * max(0.0, float(opened[ci])))
        edge_left = np.maximum(0.0, u.copy())
        arcs_by_k = {}
        valid_routes = {}
        warning = self.field(o, 'warning.score', np.zeros(len(self.l.get('warning_units', [])))).reshape(-1)
        warning_map = {tuple(key): float(np.clip(warning[i], 0.0, 1.0))
                       for i, key in enumerate(self.l.get('warning_units', [])) if i < warning.size}
        for r in self.routes:
            i, k, es, cps = r['slot'], r['k'], r['edges'], r['cps']
            if i >= mask.size or not mask[i] or any(prohibited[e, k] for e in es):
                continue
            lead = float(sum(tau[e] for e in es))
            if week + lead > self.T + 1:
                continue
            elapsed = 0.0
            is_pending = False
            for e in es:
                if pending.get((e, k), self.T + 1) <= week + elapsed:
                    is_pending = True
                    break
                elapsed += tau[e]
            if is_pending:
                continue
            cap = min((max(0.0, float(u[e])) for e in es), default=0.0)
            for cp in cps:
                cap = min(cap, cp_left.get((cp, self.pools[k]), 0.0))
            if cap <= 0:
                continue
            freight = sum(float(c[e]) + float(tariff[e, k]) * float(self.values[k]) for e in es)
            risk = max([warning_map.get(('chokepoint', cp), 0.0) for cp in cps] or [0.0])
            freight = max(0.0, freight) * (1.0 + 0.25 * risk)
            # A small time cost breaks ties in favor of faster routes.
            cost = freight + 0.002 * float(self.values[k]) * lead
            rr = dict(r, lead=lead, cap=cap, cost=cost)
            valid_routes[i] = rr
            arcs_by_k.setdefault(k, {}).setdefault(r['src'], []).append(rr)

        # Each currently stocked source gets a shortest-path tree per good.
        # Generate sink plans using those trees; only the first action is sent.
        candidates = []
        sinks_by_k = {}
        for di, (sink, k) in enumerate(self.demands):
            sinks_by_k.setdefault(k, []).append((di, sink))
        for (src, k), qty in list(available.items()):
            if qty <= 0 or k not in sinks_by_k:
                continue
            adj = arcs_by_k.get(k, {})
            dist = {src: 0.0}
            prev = {}
            heap = [(0.0, src)]
            while heap:
                d, node = heapq.heappop(heap)
                if d > dist.get(node, float('inf')) + 1e-10:
                    continue
                for r in adj.get(node, ()):
                    nd = d + r['cost']
                    if nd < dist.get(r['dst'], float('inf')) - 1e-10:
                        dist[r['dst']] = nd
                        prev[r['dst']] = (node, r)
                        heapq.heappush(heap, (nd, r['dst']))
            for di, sink in sinks_by_k[k]:
                if sink == src or sink not in dist:
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
                physical_lead = sum(r['lead'] for r in path)
                # Allow for weekly decisions between successive route legs.
                effective_lead = physical_lead + max(0, len(path) - 1)
                arrival = week + int(np.ceil(effective_lead))
                if arrival > self.T:
                    continue
                horizon = max(1, arrival - week + 2)
                cutoff = week + horizon - 1
                target = demand_total(di, horizon)
                gap = max(0.0, target - covered((sink, k), cutoff))
                if gap <= 1e-9:
                    continue
                pi = max(1.0, float(self.pi[di]))
                urgency = 0.45 + 0.55 * min(1.0, gap / max(target, 1e-9))
                score = dist[sink] / (pi * urgency)
                candidates.append((score, effective_lead, path[0]['slot'], 'sink', di, path, gap, arrival))

        # Limited fallback replenishment for non-demand intermediate nodes,
        # especially goods used in production chains.
        for i, r in valid_routes.items():
            p = (r['dst'], r['k'])
            if p in self.demand_index:
                continue
            # Sink-directed plans take precedence when this commodity can
            # continue from this destination to an actual demand sink.
            if any(sink != r['dst'] and sink in self._reachable_nodes(r['dst'], r['k'], arcs_by_k)
                   for _, sink in sinks_by_k.get(r['k'], [])):
                continue
            priority = max(1.0, self.node_priority.get(p, self.max_pi * 0.20))
            score = r['cost'] / priority
            candidates.append((score, r['lead'], i, 'buffer', -1, [r], 0.0, week + int(np.ceil(r['lead']))))

        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        flows = np.zeros(self.nslots, dtype=float)
        reservations = {}
        for _, effective_lead, slot, kind, di, path, initial_gap, sink_arrival in candidates:
            first = path[0]
            r = valid_routes.get(slot)
            if r is None:
                continue
            k, src, dst = first['k'], first['src'], first['dst']
            src_pair, dst_pair = (src, k), (dst, k)
            q = min(float(first['cap']), max(0.0, available.get(src_pair, 0.0)))
            for e in first['edges']:
                q = min(q, max(0.0, float(edge_left[e])))
            for cp in first['cps']:
                q = min(q, cp_left.get((cp, self.pools[k]), 0.0))

            if kind == 'sink':
                sink = self.demands[di][0]
                horizon = max(1, int(np.ceil(effective_lead)) + 2)
                cutoff = week + horizon - 1
                target = demand_total(di, horizon)
                reserved = sum(amount for arrival, amount in reservations.get((sink, k), []) if arrival <= cutoff)
                unmet = max(0.0, target - covered((sink, k), cutoff) - reserved)
                q = min(q, unmet)
                if q > 0:
                    reservations.setdefault((sink, k), []).append((sink_arrival, q))
            else:
                duration = min(max(2.0, first['lead'] + 2.0), max(1.0, self.T - week + 1))
                cutoff = week + int(np.ceil(duration))
                target = max(0.0, self.node_rate.get(dst_pair, 0.0)) * duration
                q = min(q, max(0.0, target - covered(dst_pair, cutoff)))

            if q <= 1e-9 or not np.isfinite(q):
                continue
            flows[slot] += q
            available[src_pair] = max(0.0, available.get(src_pair, 0.0) - q)
            for e in first['edges']:
                edge_left[e] = max(0.0, edge_left[e] - q)
            for cp in first['cps']:
                key = (cp, self.pools[k])
                cp_left[key] = max(0.0, cp_left.get(key, 0.0) - q)
            add_projection(dst_pair, week + int(np.ceil(first['lead'])), q)

        return {'flows': flows}

    @staticmethod
    def _reachable_nodes(src, k, arcs_by_k):
        adj = arcs_by_k.get(k, {})
        seen = {src}
        stack = [src]
        while stack:
            node = stack.pop()
            for r in adj.get(node, ()):
                if r['dst'] not in seen:
                    seen.add(r['dst'])
                    stack.append(r['dst'])
        return seen