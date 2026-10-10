# 0.43541571048092975
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
        self.nodes_type = s['nodes'].get('type', [])
        self.nodes_region = s['nodes'].get('region', [])
        self.stock_slots = [tuple(map(int, x)) for x in l['stock_slots']]
        self.supply_slots = [tuple(map(int, x)) for x in l['supply_slots']]
        self.demands = [tuple(map(int, x)) for x in l['demands']]
        self.demand_index = {p: i for i, p in enumerate(self.demands)}
        self.cp_index = {int(n): i for i, n in enumerate(l['chokepoints'])}
        self.pi = np.asarray(s.get('sinks', {}).get('pi', np.ones(len(self.demands))), dtype=float)
        self.max_pi = max(1.0, float(np.max(self.pi)) if self.pi.size else 1.0)

        self.routes = []
        slots = s['action_slots']
        grouped = {}
        for i, (edge, commodity, lane) in enumerate(zip(slots['edge'], slots['k'], slots['lane'])):
            edge, commodity = int(edge), int(commodity)
            lane = -1 if lane is None else int(lane)
            if lane < 0:
                route_edges = [edge]
                cps = []
                alt = edges['alt_of'][edge]
            else:
                route_edges = [int(e) for e in s['lanes']['edges'][lane]]
                cps = [int(cp) for cp in s['lanes']['chokepoints'][lane]]
                alt = s['lanes']['alt_of'][lane]
            if not route_edges:
                continue
            src = int(self.tail[route_edges[0]])
            dst = int(self.head[route_edges[-1]])
            route = {
                'slot': i, 'k': commodity, 'edges': route_edges, 'cps': cps,
                'src': src, 'dst': dst, 'alt': alt
            }
            self.routes.append(route)
            cap = min((self.u0[e] for e in route_edges), default=0.0)
            grouped.setdefault((src, commodity, dst), []).append((alt, cap))

        # Approximate sustainable node throughput without counting explicit
        # alternate routes between the same endpoints twice.
        incoming, outgoing = {}, {}
        for (src, k, dst), choices in grouped.items():
            ordinary = [cap for alt, cap in choices if alt is None]
            cap = sum(ordinary) if ordinary else max((x[1] for x in choices), default=0.0)
            incoming[(dst, k)] = incoming.get((dst, k), 0.0) + cap
            outgoing[(src, k)] = outgoing.get((src, k), 0.0) + cap
        self.node_rate = {}
        for p in set(incoming) | set(outgoing):
            a, b = incoming.get(p), outgoing.get(p)
            self.node_rate[p] = min(a, b) if a is not None and b is not None else max(a or 0.0, b or 0.0)

        # Back-propagated sink value is used only for the conservative
        # intermediate-stock fallback, not as a substitute for sink demand.
        reverse = {}
        for r in self.routes:
            reverse.setdefault((r['dst'], r['k']), set()).add(r['src'])
        self.node_priority = {}
        for di, (sink, k) in enumerate(self.demands):
            penalty = max(1.0, float(self.pi[di]) if di < self.pi.size else 1.0)
            queue = [(int(sink), 0)]
            seen = {int(sink)}
            pos = 0
            while pos < len(queue):
                node, distance = queue[pos]
                pos += 1
                key = (node, int(k))
                self.node_priority[key] = max(
                    self.node_priority.get(key, 0.0), penalty / (1.0 + 0.12 * distance)
                )
                for prev in reverse.get(key, ()):
                    if prev not in seen:
                        seen.add(prev)
                        queue.append((prev, distance + 1))

    @staticmethod
    def _field(o, name, fallback):
        fb = np.asarray(fallback)
        if name not in o:
            return fb.copy()
        x = np.asarray(o[name])
        mask = o.get(name + '.observed')
        if mask is not None and np.shape(mask) == x.shape:
            return np.where(mask, x, fb)
        return x.copy()

    @staticmethod
    def _live(o, name, size):
        x = np.asarray(o.get(name, np.zeros(size))).reshape(-1)
        m = np.asarray(o.get(name + '.observed', np.ones(x.size)), dtype=bool).reshape(-1)
        return x, m

    def act(self, o):
        week = int(np.asarray(o['week']).reshape(-1)[0])
        nedge, nk = len(self.tail), len(self.values)
        nslot = len(self.s['action_slots']['edge'])
        ncp = len(self.cp_index)
        u = self._field(o, 'graph_now.u', self.u0).astype(float)
        c = self._field(o, 'graph_now.c', self.c0).astype(float)
        tau = self._field(o, 'graph_now.tau', self.tau0).astype(float)
        tariff = self._field(o, 'graph_now.tariff', np.zeros((nedge, nk))).astype(float)
        prohibited = self._field(o, 'graph_now.prohibited', np.zeros((nedge, nk))).astype(bool)
        prohibited_observed = np.asarray(
            o.get('graph_now.prohibited.observed', np.ones_like(prohibited)), dtype=bool
        )
        opened = self._field(o, 'graph_now.open', np.ones(ncp)).astype(float)
        ktb = self._field(o, 'graph_now.kappa.tb', np.full(ncp, np.inf)).astype(float)
        kct = self._field(o, 'graph_now.kappa.ct', np.full(ncp, np.inf)).astype(float)
        mask = np.asarray(o.get('action_mask', np.ones(nslot)), dtype=bool).reshape(-1)
        mask_seen = bool(np.asarray(o.get('action_mask.observed', [1])).reshape(-1)[0])
        slot_mask = self._field(o, 'slot_mask', np.zeros(nslot)).astype(bool).reshape(-1)

        # Current inventory is dispatchable; future arrivals are tracked for
        # demand coverage, but cannot be dispatched before they arrive.
        stock_vec = self._field(o, 'stock.qty', np.zeros(len(self.stock_slots))).astype(float)
        stock_left = {}
        available = {}
        for i, p in enumerate(self.stock_slots):
            q = max(0.0, float(stock_vec[i])) if i < stock_vec.size else 0.0
            stock_left[p] = stock_left.get(p, 0.0) + q
            available[p] = available.get(p, 0.0) + q
        supply = self._field(o, 'graph_now.supply.avail', np.zeros(len(self.supply_slots))).astype(float)
        for i, p in enumerate(self.supply_slots):
            if i < supply.size:
                q = max(0.0, float(supply[i]))
                available[p] = available.get(p, 0.0) + q

        projected = {}

        def add_projected(p, arrival, qty):
            if qty > 0 and np.isfinite(qty):
                projected.setdefault(p, []).append((int(arrival), float(qty)))

        # Existing shipments, including the remaining part of a multi-edge lane.
        pqty = np.asarray(o.get('pipeline.qty', [])).reshape(-1)
        plive = np.asarray(o.get('pipeline.qty.observed', np.ones(pqty.size)), dtype=bool).reshape(-1)
        pedge = np.asarray(o.get('pipeline.edge', np.zeros(pqty.size)), dtype=int).reshape(-1)
        pk = np.asarray(o.get('pipeline.k', np.zeros(pqty.size)), dtype=int).reshape(-1)
        plane = np.asarray(o.get('pipeline.lane', np.full(pqty.size, -1)), dtype=int).reshape(-1)
        plane_live = np.asarray(o.get('pipeline.lane.observed', np.ones(pqty.size)), dtype=bool).reshape(-1)
        parr = np.asarray(o.get('pipeline.arrival_week', np.full(pqty.size, week)), dtype=int).reshape(-1)
        lanes = self.s.get('lanes', {}).get('edges', [])
        for j in range(min(pqty.size, pedge.size, pk.size, parr.size)):
            if j >= plive.size or not plive[j] or pqty[j] <= 0:
                continue
            edge = int(pedge[j])
            if edge < 0 or edge >= nedge:
                continue
            lane = int(plane[j]) if j < plane.size and j < plane_live.size and plane_live[j] else -1
            dst = int(self.head[edge])
            arrival = int(parr[j])
            if 0 <= lane < len(lanes):
                les = [int(e) for e in lanes[lane]]
                if les:
                    dst = int(self.head[les[-1]])
                    if edge in les:
                        pos = les.index(edge)
                        arrival += int(np.ceil(sum(max(1.0, float(tau[e])) for e in les[pos + 1:])))
            add_projected((dst, int(pk[j])), max(week, arrival), pqty[j])

        wqty, wlive = self._live(o, 'wip.qty', 0)
        wnode = np.asarray(o.get('wip.node', np.zeros(wqty.size)), dtype=int).reshape(-1)
        wk = np.asarray(o.get('wip.k', np.zeros(wqty.size)), dtype=int).reshape(-1)
        wout = np.asarray(o.get('wip.out_week', np.full(wqty.size, week)), dtype=int).reshape(-1)
        for j in range(min(wqty.size, wnode.size, wk.size, wout.size)):
            if j < wlive.size and wlive[j] and wqty[j] > 0:
                add_projected((int(wnode[j]), int(wk[j])), max(week, int(wout[j])), wqty[j])

        if 'queue_lots.qty' in o:
            lots = np.asarray(o['queue_lots.qty'], dtype=float)
            lot_live = np.asarray(o.get('queue_lots.qty.observed', np.ones(lots.shape)), dtype=bool)
            for row, key in enumerate(self.l.get('lot_keys', [])[:lots.shape[0]]):
                if len(key) < 4:
                    continue
                _, commodity, lane, next_edge = key
                commodity, next_edge = int(commodity), int(next_edge)
                if not 0 <= next_edge < nedge:
                    continue
                lane = -1 if lane is None else int(lane)
                if 0 <= lane < len(lanes) and lanes[lane]:
                    les = [int(e) for e in lanes[lane]]
                    dst = int(self.head[les[-1]])
                    rem = sum(max(1.0, float(tau[e])) for e in les[les.index(next_edge):]) if next_edge in les else max(1.0, float(tau[next_edge]))
                else:
                    dst, rem = int(self.head[next_edge]), max(1.0, float(tau[next_edge]))
                for col in range(lots.shape[1]):
                    if row < lot_live.shape[0] and col < lot_live.shape[1] and lot_live[row, col] and lots[row, col] > 0:
                        add_projected((dst, commodity), max(week, col + 1) + int(np.ceil(rem)), lots[row, col])

        # Announced prohibitions let the policy avoid starting a route that is
        # expected to become illegal before the shipment reaches that edge.
        pending = {}
        pe, pe_live = self._live(o, 'pending_prohibitions.edge', 0)
        pkp = np.asarray(o.get('pending_prohibitions.k', np.zeros(pe.size)), dtype=int).reshape(-1)
        pwe = np.asarray(o.get('pending_prohibitions.effective_week', np.full(pe.size, self.T + 1)), dtype=int).reshape(-1)
        for j in range(min(pe.size, pkp.size, pwe.size)):
            if j < pe_live.size and pe_live[j]:
                key = (int(pe[j]), int(pkp[j]))
                pending[key] = min(pending.get(key, self.T + 1), int(pwe[j]))

        warnings = {}
        warning = self._field(o, 'warning.score', np.zeros(len(self.l.get('warning_units', [])))).astype(float)
        for key, value in zip(self.l.get('warning_units', []), warning):
            try:
                warnings[tuple(key)] = float(np.clip(value, 0.0, 1.0))
            except Exception:
                pass

        forecast = np.asarray(o.get('demand_forecast.qty', np.zeros((len(self.demands), 1))), dtype=float)
        backlog = np.asarray(o.get('backlog.qty', np.zeros(len(self.demands))), dtype=float).reshape(-1)

        def projected_at(p, cutoff):
            current = stock_left.get(p, 0.0) if cutoff >= week else 0.0
            return current + sum(q for arrival, q in projected.get(p, ()) if arrival <= cutoff)

        def demand_total(di, horizon):
            base = max(0.0, float(backlog[di])) if di < backlog.size else 0.0
            if forecast.ndim != 2 or di >= forecast.shape[0] or forecast.shape[1] == 0:
                return base
            h = max(1, int(horizon))
            row = forecast[di]
            used = min(h, row.size)
            total = base + float(np.sum(row[:used]))
            if h > used:
                tail = row[-min(3, row.size):]
                total += (h - used) * max(0.0, float(np.mean(tail)))
            return total

        def sink_need(di, total_lead, extra_horizon=0):
            sink, k = self.demands[di]
            horizon = min(max(1, self.T - week + 1), int(np.ceil(total_lead)) + 2 + int(extra_horizon))
            cutoff = week + horizon - 1
            target = demand_total(di, horizon)
            return max(0.0, target - projected_at((sink, k), cutoff)), cutoff

        # Chokepoint and edge capacity is reserved once across all chosen slots.
        cp_left = {}
        for cp, ci in self.cp_index.items():
            for pool, caps in (('tb', ktb), ('ct', kct)):
                cap = float(caps[ci]) if ci < caps.size else np.inf
                cp_left[(cp, pool)] = max(0.0, cap * max(0.0, float(opened[ci])))
        edge_left = np.maximum(0.0, u.copy())
        arcs_by_k = {}
        route_by_slot = {}
        usable = []
        for r in self.routes:
            i, k, es, cps = r['slot'], r['k'], r['edges'], r['cps']
            if i >= nslot or not es or any(e < 0 or e >= nedge for e in es):
                continue
            legal = True
            if mask_seen and (i >= mask.size or not mask[i]):
                legal = False
            if i < slot_mask.size and slot_mask[i]:
                legal = False
            for e in es:
                if k < prohibited.shape[1] and e < prohibited.shape[0] and prohibited_observed[e, k] and prohibited[e, k]:
                    legal = False
                elapsed = sum(max(1.0, float(tau[x])) for x in es[:es.index(e)])
                if pending.get((e, k), self.T + 1) <= week + elapsed:
                    legal = False
            if not legal:
                continue
            lead = sum(max(1.0, float(tau[e])) for e in es)
            if week + lead > self.T + 1:
                continue
            cap = min((max(0.0, float(u[e])) for e in es), default=0.0)
            for cp in cps:
                ci = self.cp_index.get(cp)
                if ci is not None:
                    cap = min(cap, cp_left.get((cp, self.pools[k]), 0.0))
            if cap <= 0:
                continue
            cp_warn = max((warnings.get(('chokepoint', cp), 0.0) for cp in cps), default=0.0)
            region_warn = 0.0
            if self.nodes_region and 0 <= r['src'] < len(self.nodes_region):
                region_warn = warnings.get(('region', int(self.nodes_region[r['src']])), 0.0)
            risk = max(cp_warn, region_warn)
            freight = 0.0
            for e in es:
                rate = float(c[e])
                if e < tariff.shape[0] and k < tariff.shape[1]:
                    rate += float(tariff[e, k]) * float(self.values[k])
                freight += rate
            # A mild warning penalty prefers a viable safer alternative, while
            # preserving the option to use the warned route if it is cheapest.
            risk_cost = risk * (0.12 * max(1.0, freight) + 0.001 * float(self.values[k]))
            weight = max(0.0, freight + risk_cost) + 0.002 * float(self.values[k]) * lead
            r['lead'] = lead
            r['cap'] = cap
            r['cost'] = weight
            r['risk'] = risk
            usable.append(r)
            route_by_slot[i] = r
            arcs_by_k.setdefault(k, {}).setdefault(r['src'], []).append(r)

        # Compute current least-cost paths from stocked/supplied nodes to each
        # sink. Candidate actions send only the first route; its full-path
        # arrival is reserved against the sink's forecast to prevent duplicates.
        candidates = []
        source_keys = [(p, q) for p, q in available.items() if q > 1e-9]
        for (src, k), _qty in source_keys:
            adj = arcs_by_k.get(k, {})
            if not adj:
                continue
            dist = {src: 0.0}
            total_time = {src: 0.0}
            prev = {}
            heap = [(0.0, src)]
            while heap:
                d, node = heapq.heappop(heap)
                if d > dist.get(node, float('inf')) + 1e-10:
                    continue
                for r in adj.get(node, ()):
                    nd = d + r['cost']
                    if nd + 1e-10 < dist.get(r['dst'], float('inf')):
                        dist[r['dst']] = nd
                        total_time[r['dst']] = total_time[node] + r['lead']
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
                if week + lead > self.T + 1:
                    continue
                path_risk = max((r['risk'] for r in path), default=0.0)
                need, _ = sink_need(di, lead, int(round(2.0 * path_risk)))
                if need <= 1e-9:
                    continue
                pi = max(1.0, float(self.pi[di]) if di < self.pi.size else 1.0)
                # Cost per unit of avoided shortage, with a small preference
                # for earlier delivery when alternatives have similar cost.
                score = dist[sink] / pi + 0.006 * lead
                candidates.append((score, lead, 'sink', path[0], di, path, path_risk))

        # A restrained buffer fallback moves same-commodity stock toward
        # useful transfer/production nodes not directly represented as sinks.
        for r in usable:
            if (r['dst'], r['k']) in self.demand_index:
                continue
            p = (r['dst'], r['k'])
            rate = max(0.0, float(self.node_rate.get(p, 0.0)))
            priority = max(1.0, self.node_priority.get(p, self.max_pi * 0.18))
            if rate <= 0:
                continue
            duration = min(4.0, max(2.0, r['lead'] + 1.0), max(1.0, self.T - week + 1))
            target = rate * duration
            cutoff = week + int(np.ceil(duration))
            need = max(0.0, target - projected_at(p, cutoff))
            if need <= 1e-9:
                continue
            score = (r['cost'] + 0.002 * float(self.values[r['k']]) * r['lead']) / (priority * 0.55)
            candidates.append((score, r['lead'], 'buffer', r, p, [r], r['risk']))

        flows = np.zeros(nslot, dtype=float)
        candidates.sort(key=lambda x: (x[0], x[1], x[3]['slot']))
        for _, lead, kind, first, target, path, risk in candidates:
            r = first
            i, k = r['slot'], r['k']
            src_key = (r['src'], k)
            dst_key = (r['dst'], k)
            if kind == 'sink':
                di = int(target)
                need, _ = sink_need(di, lead, int(round(2.0 * risk)))
                arrival_sink = week + int(np.ceil(lead))
                dest_projection = (self.demands[di][0], k)
            else:
                dest_projection = target
                duration = min(4.0, max(2.0, r['lead'] + 1.0), max(1.0, self.T - week + 1))
                cutoff = week + int(np.ceil(duration))
                desired = max(0.0, float(self.node_rate.get(dest_projection, 0.0))) * duration
                need = max(0.0, desired - projected_at(dest_projection, cutoff))
                arrival_sink = week + int(np.ceil(r['lead']))
            have = max(0.0, available.get(src_key, 0.0))
            q = min(float(r['cap']), need, have)
            for e in r['edges']:
                q = min(q, max(0.0, float(edge_left[e])))
            for cp in r['cps']:
                if cp in self.cp_index:
                    q = min(q, cp_left.get((cp, self.pools[k]), 0.0))
            if q <= 1e-9 or not np.isfinite(q):
                continue
            flows[i] += q
            available[src_key] = max(0.0, have - q)
            used_stock = min(q, stock_left.get(src_key, 0.0))
            stock_left[src_key] = max(0.0, stock_left.get(src_key, 0.0) - used_stock)
            for e in r['edges']:
                edge_left[e] = max(0.0, edge_left[e] - q)
            for cp in r['cps']:
                if cp in self.cp_index:
                    key = (cp, self.pools[k])
                    cp_left[key] = max(0.0, cp_left.get(key, 0.0) - q)
            arrival_first = week + int(np.ceil(r['lead']))
            add_projected(dst_key, arrival_first, q)
            if kind == 'sink' and dst_key != dest_projection:
                add_projected(dest_projection, arrival_sink, q)

        return {'flows': flows}
