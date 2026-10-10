# 0.3078788045515111
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
        self.node_types = s['nodes'].get('type', [])
        self.node_regions = s['nodes'].get('region', [])
        self.stock_slots = [tuple(map(int, x)) for x in l['stock_slots']]
        self.supply_slots = [tuple(map(int, x)) for x in l['supply_slots']]
        self.demands = [tuple(map(int, x)) for x in l['demands']]
        self.demand_index = {p: i for i, p in enumerate(self.demands)}
        self.cp_index = {int(n): i for i, n in enumerate(l['chokepoints'])}
        self.pi = np.asarray(s.get('sinks', {}).get('pi', np.ones(len(self.demands))), dtype=float)
        self.max_pi = max(1.0, float(np.max(self.pi)) if self.pi.size else 1.0)
        self.lanes = s.get('lanes', {}).get('edges', [])
        self.routes = []
        slots = s['action_slots']
        grouped = {}
        reverse = {}

        for i, (edge, commodity, lane) in enumerate(zip(slots['edge'], slots['k'], slots['lane'])):
            edge, commodity = int(edge), int(commodity)
            lane = -1 if lane is None else int(lane)
            route_edges = [edge] if lane < 0 else [int(x) for x in self.lanes[lane]]
            cps = [] if lane < 0 else [int(x) for x in s['lanes']['chokepoints'][lane]]
            if not route_edges:
                self.routes.append(None)
                continue
            src = int(self.tail[route_edges[0]])
            dst = int(self.head[route_edges[-1]])
            alt = edges['alt_of'][edge] if lane < 0 else s['lanes']['alt_of'][lane]
            r = {'slot': i, 'k': commodity, 'edges': route_edges, 'cps': cps,
                 'src': src, 'dst': dst, 'alt': alt}
            self.routes.append(r)
            cap = min((self.u0[e] for e in route_edges), default=0.0)
            grouped.setdefault((src, commodity, dst), []).append((alt, cap))
            reverse.setdefault((dst, commodity), set()).add(src)

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

        self.node_priority = {}
        for di, (sink, k) in enumerate(self.demands):
            penalty = max(1.0, float(self.pi[di]) if di < self.pi.size else 1.0)
            queue = [(int(sink), 0)]
            seen = {int(sink)}
            pos = 0
            while pos < len(queue):
                node, distance = queue[pos]
                pos += 1
                p = (node, int(k))
                self.node_priority[p] = max(
                    self.node_priority.get(p, 0.0), penalty / (1.0 + 0.15 * distance)
                )
                for prev in reverse.get(p, ()):
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
    def _live(o, name):
        x = np.asarray(o.get(name, [])).reshape(-1)
        m = np.asarray(o.get(name + '.observed', np.ones(x.size)), dtype=bool).reshape(-1)
        return x, m

    def act(self, o):
        week = int(np.asarray(o['week']).reshape(-1)[0])
        nedge, nk = len(self.tail), len(self.values)
        nslot = len(self.s['action_slots']['edge'])
        ncp = len(self.cp_index)
        u = self._field(o, 'graph_now.u', self.u0).astype(float)
        c = self._field(o, 'graph_now.c', self.c0).astype(float)
        tau = np.maximum(1.0, self._field(o, 'graph_now.tau', self.tau0).astype(float))
        tariff = self._field(o, 'graph_now.tariff', np.zeros((nedge, nk))).astype(float)
        prohibited = self._field(o, 'graph_now.prohibited', np.zeros((nedge, nk))).astype(bool)
        prohibited_seen = np.asarray(
            o.get('graph_now.prohibited.observed', np.ones_like(prohibited)), dtype=bool
        )
        opened = self._field(o, 'graph_now.open', np.ones(ncp)).astype(float)
        ktb = self._field(o, 'graph_now.kappa.tb', np.full(ncp, np.inf)).astype(float)
        kct = self._field(o, 'graph_now.kappa.ct', np.full(ncp, np.inf)).astype(float)
        mask = np.asarray(o.get('action_mask', np.ones(nslot)), dtype=bool).reshape(-1)
        mask_seen = bool(np.asarray(o.get('action_mask.observed', [1])).reshape(-1)[0])
        slot_bad = np.asarray(o.get('slot_mask', np.zeros(nslot)), dtype=bool).reshape(-1)
        slot_bad_seen = np.asarray(
            o.get('slot_mask.observed', np.ones(nslot)), dtype=bool
        ).reshape(-1)

        stock_vec = self._field(o, 'stock.qty', np.zeros(len(self.stock_slots))).astype(float)
        stock_left = {}
        available = {}
        projected = {}

        def add_projected(p, arrival, qty):
            if qty > 0 and np.isfinite(qty):
                projected.setdefault(p, []).append((int(arrival), float(qty)))

        for i, p in enumerate(self.stock_slots):
            q = max(0.0, float(stock_vec[i])) if i < stock_vec.size else 0.0
            stock_left[p] = stock_left.get(p, 0.0) + q
            available[p] = available.get(p, 0.0) + q
        supply = self._field(o, 'graph_now.supply.avail', np.zeros(len(self.supply_slots))).astype(float)
        for i, p in enumerate(self.supply_slots):
            if i < supply.size:
                available[p] = available.get(p, 0.0) + max(0.0, float(supply[i]))

        # Existing shipments, accounting for the uncompleted part of a lane.
        pqty = np.asarray(o.get('pipeline.qty', [])).reshape(-1)
        plive = np.asarray(o.get('pipeline.qty.observed', np.ones(pqty.size)), dtype=bool).reshape(-1)
        pedge = np.asarray(o.get('pipeline.edge', np.zeros(pqty.size)), dtype=int).reshape(-1)
        pk = np.asarray(o.get('pipeline.k', np.zeros(pqty.size)), dtype=int).reshape(-1)
        plane = np.asarray(o.get('pipeline.lane', np.full(pqty.size, -1)), dtype=int).reshape(-1)
        plane_live = np.asarray(o.get('pipeline.lane.observed', np.ones(pqty.size)), dtype=bool).reshape(-1)
        parr = np.asarray(o.get('pipeline.arrival_week', np.full(pqty.size, week)), dtype=int).reshape(-1)
        for j in range(min(pqty.size, pedge.size, pk.size, parr.size)):
            if j >= plive.size or not plive[j] or pqty[j] <= 0:
                continue
            edge = int(pedge[j])
            if not 0 <= edge < nedge:
                continue
            lane = int(plane[j]) if j < plane.size and j < plane_live.size and plane_live[j] else -1
            dst, arrival = int(self.head[edge]), int(parr[j])
            if 0 <= lane < len(self.lanes):
                les = [int(e) for e in self.lanes[lane]]
                if les:
                    dst = int(self.head[les[-1]])
                    if edge in les:
                        pos = les.index(edge)
                        arrival += int(np.ceil(sum(tau[e] for e in les[pos + 1:])))
            add_projected((dst, int(pk[j])), max(week, arrival), float(pqty[j]))

        # Production WIP becomes stock at its stated output week.
        wqty, wlive = self._live(o, 'wip.qty')
        wnode = np.asarray(o.get('wip.node', np.zeros(wqty.size)), dtype=int).reshape(-1)
        wk = np.asarray(o.get('wip.k', np.zeros(wqty.size)), dtype=int).reshape(-1)
        wout = np.asarray(o.get('wip.out_week', np.full(wqty.size, week)), dtype=int).reshape(-1)
        for j in range(min(wqty.size, wnode.size, wk.size, wout.size)):
            if j < wlive.size and wlive[j] and wqty[j] > 0:
                add_projected((int(wnode[j]), int(wk[j])), max(week, int(wout[j])), float(wqty[j]))

        # Lots waiting at chokepoints are treated as future arrivals downstream.
        closure_end = {}
        ce, ce_live = self._live(o, 'closure_end.chokepoint')
        cend = np.asarray(o.get('closure_end.end_week', np.zeros(ce.size)), dtype=int).reshape(-1)
        cend_live = np.asarray(o.get('closure_end.end_week.observed', np.zeros(ce.size)), dtype=bool).reshape(-1)
        for j in range(min(ce.size, cend.size, ce_live.size, cend_live.size)):
            if ce_live[j] and cend_live[j]:
                cp = int(ce[j])
                closure_end[cp] = max(closure_end.get(cp, week), int(cend[j]) + 1)
        lots = np.asarray(o.get('queue_lots.qty', np.zeros((0, 0))), dtype=float)
        lots_live = np.asarray(o.get('queue_lots.qty.observed', np.ones(lots.shape)), dtype=bool)
        for row, key in enumerate(self.l.get('lot_keys', [])[:lots.shape[0]]):
            if len(key) < 4:
                continue
            cp, k, lane, next_edge = key
            next_edge = int(next_edge)
            if not 0 <= next_edge < nedge:
                continue
            lane = -1 if lane is None else int(lane)
            if 0 <= lane < len(self.lanes) and self.lanes[lane]:
                les = [int(e) for e in self.lanes[lane]]
                dst = int(self.head[les[-1]])
                rem = sum(tau[e] for e in les[les.index(next_edge):]) if next_edge in les else tau[next_edge]
            else:
                dst, rem = int(self.head[next_edge]), tau[next_edge]
            start = max(week, closure_end.get(int(cp), week))
            ci = self.cp_index.get(int(cp))
            if ci is not None and opened[ci] <= 0.001 and int(cp) not in closure_end:
                start += 2
            if row >= lots_live.shape[0]:
                continue
            for col in range(min(lots.shape[1], lots_live.shape[1])):
                if lots_live[row, col] and lots[row, col] > 0:
                    arrival = start + int(np.ceil(rem))
                    add_projected((dst, int(k)), arrival, float(lots[row, col]))

        pending = {}
        pe, pe_live = self._live(o, 'pending_prohibitions.edge')
        pkp = np.asarray(o.get('pending_prohibitions.k', np.zeros(pe.size)), dtype=int).reshape(-1)
        pwe = np.asarray(o.get('pending_prohibitions.effective_week', np.full(pe.size, self.T + 1)), dtype=int).reshape(-1)
        for j in range(min(pe.size, pkp.size, pwe.size, pe_live.size)):
            if pe_live[j]:
                key = (int(pe[j]), int(pkp[j]))
                pending[key] = min(pending.get(key, self.T + 1), int(pwe[j]))

        warning = self._field(o, 'warning.score', np.zeros(len(self.l.get('warning_units', [])))).astype(float)
        warnings = {}
        for key, value in zip(self.l.get('warning_units', []), warning):
            try:
                warnings[tuple(key)] = float(np.clip(value, 0.0, 1.0))
            except Exception:
                pass

        forecast = np.asarray(o.get('demand_forecast.qty', np.zeros((len(self.demands), 1))), dtype=float)
        backlog = np.asarray(o.get('backlog.qty', np.zeros(len(self.demands))), dtype=float).reshape(-1)

        def projected_qty(p, cutoff):
            current = stock_left.get(p, 0.0) if cutoff >= week else 0.0
            return current + sum(q for arrival, q in projected.get(p, ()) if arrival <= cutoff)

        def demand_total(di, horizon):
            total = max(0.0, float(backlog[di])) if di < backlog.size else 0.0
            if forecast.ndim != 2 or di >= forecast.shape[0] or forecast.shape[1] == 0:
                return total
            row = forecast[di]
            used = min(max(0, int(horizon)), row.size)
            if used:
                total += float(np.sum(row[:used]))
            if horizon > used and row.size:
                tail = row[-min(3, row.size):]
                total += (horizon - used) * max(0.0, float(np.mean(tail)))
            return total

        def sink_need(di, lead):
            horizon = min(max(0, self.T - week + 1), max(1, int(np.ceil(lead)) + 2))
            if horizon <= 0:
                return 0.0, week, 0.0
            cutoff = min(self.T, week + horizon - 1)
            target = demand_total(di, horizon)
            sink, k = self.demands[di]
            need = max(0.0, target - projected_qty((sink, k), cutoff))
            return need, cutoff, target

        # Reserve shared edge and chokepoint capacities across all action slots.
        edge_left = np.maximum(0.0, u.copy())
        cp_left = {}
        for cp, ci in self.cp_index.items():
            for pool, caps in (('tb', ktb), ('ct', kct)):
                cap = float(caps[ci]) if ci < caps.size else np.inf
                cp_left[(cp, pool)] = max(0.0, cap * max(0.0, float(opened[ci])))

        usable = []
        arcs_by_k = {}
        for r in self.routes:
            if r is None:
                continue
            i, k, es, cps = r['slot'], r['k'], r['edges'], r['cps']
            if not es or i >= nslot:
                continue
            if mask_seen and (i >= mask.size or not mask[i]):
                continue
            if i < slot_bad.size and i < slot_bad_seen.size and slot_bad_seen[i] and slot_bad[i]:
                continue
            legal = True
            elapsed = 0.0
            for e in es:
                if e < 0 or e >= nedge:
                    legal = False
                    break
                if k < prohibited.shape[1] and e < prohibited.shape[0]:
                    if e < prohibited_seen.shape[0] and k < prohibited_seen.shape[1] and prohibited_seen[e, k] and prohibited[e, k]:
                        legal = False
                if pending.get((e, k), self.T + 1) <= week + elapsed:
                    legal = False
                elapsed += float(tau[e])
            if not legal:
                continue
            if any(self.cp_index.get(cp) is not None and opened[self.cp_index[cp]] <= 0.001 for cp in cps):
                continue
            lead = float(sum(tau[e] for e in es))
            if week + lead > self.T + 1:
                continue
            cap = min((max(0.0, float(u[e])) for e in es), default=0.0)
            for cp in cps:
                ci = self.cp_index.get(cp)
                if ci is not None:
                    cap = min(cap, cp_left.get((cp, self.pools[k]), 0.0))
            if cap <= 0:
                continue
            freight = 0.0
            for e in es:
                rate = float(c[e])
                if e < tariff.shape[0] and k < tariff.shape[1]:
                    rate += float(tariff[e, k]) * float(self.values[k])
                freight += rate
            cp_risk = max((warnings.get(('chokepoint', cp), 0.0) for cp in cps), default=0.0)
            region = int(self.node_regions[r['src']]) if r['src'] < len(self.node_regions) else -1
            region_risk = warnings.get(('region', region), 0.0)
            risk = max(cp_risk, region_risk)
            cost = max(0.0, freight) + 0.0015 * float(self.values[k]) * lead
            cost += risk * (0.10 * max(1.0, freight) + 0.0005 * float(self.values[k]))
            r['lead'] = lead
            r['cap'] = cap
            r['cost'] = cost
            usable.append(r)
            arcs_by_k.setdefault(k, {}).setdefault(r['src'], []).append(r)

        # For every stocked/supplied source, find least-cost legal paths to sinks.
        candidates = []
        for (src, k), qty in list(available.items()):
            if qty <= 1e-9 or k not in arcs_by_k:
                continue
            dist = {src: 0.0}
            path_time = {src: 0.0}
            prev = {}
            heap = [(0.0, src)]
            while heap:
                d, node = heapq.heappop(heap)
                if d > dist.get(node, float('inf')) + 1e-10:
                    continue
                for r in arcs_by_k[k].get(node, ()):
                    nd = d + r['cost']
                    if nd + 1e-10 < dist.get(r['dst'], float('inf')):
                        dist[r['dst']] = nd
                        path_time[r['dst']] = path_time[node] + r['lead']
                        prev[r['dst']] = (node, r)
                        heapq.heappush(heap, (nd, r['dst']))
            for di, (sink, dk) in enumerate(self.demands):
                if dk != k or sink not in dist or sink == src:
                    continue
                path = []
                node = int(sink)
                seen = set()
                while node != src and node in prev and node not in seen:
                    seen.add(node)
                    parent, route = prev[node]
                    path.append(route)
                    node = parent
                if node != src or not path:
                    continue
                path.reverse()
                lead = sum(r['lead'] for r in path)
                if week + lead > self.T + 1:
                    continue
                need, _, target = sink_need(di, lead)
                if need <= 1e-9:
                    continue
                pi = max(1.0, float(self.pi[di]) if di < self.pi.size else 1.0)
                shortage_fraction = min(1.0, need / max(1.0, target))
                score = dist[sink] / (pi * (1.0 + 0.4 * shortage_fraction)) + 0.003 * lead
                candidates.append((score, lead, di, path))

        flows = np.zeros(nslot, dtype=float)
        candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3][0]['slot']))
        for _, lead, di, path in candidates:
            first = path[0]
            i, k = first['slot'], first['k']
            src_key = (first['src'], k)
            dst_key = (first['dst'], k)
            need, _, _ = sink_need(di, lead)
            have = max(0.0, available.get(src_key, 0.0))
            q = min(first['cap'], need, have)
            for e in first['edges']:
                q = min(q, max(0.0, float(edge_left[e])))
            for cp in first['cps']:
                q = min(q, cp_left.get((cp, self.pools[k]), 0.0))
            if q <= 1e-9 or not np.isfinite(q):
                continue
            flows[i] += q
            available[src_key] = max(0.0, have - q)
            stock_left[src_key] = max(0.0, stock_left.get(src_key, 0.0) - min(q, stock_left.get(src_key, 0.0)))
            for e in first['edges']:
                edge_left[e] = max(0.0, edge_left[e] - q)
            for cp in first['cps']:
                key = (cp, self.pools[k])
                if key in cp_left:
                    cp_left[key] = max(0.0, cp_left[key] - q)
            first_arrival = week + int(np.ceil(first['lead']))
            add_projected(dst_key, first_arrival, q)
            sink_key = self.demands[di]
            if dst_key != sink_key:
                add_projected(sink_key, week + int(np.ceil(lead)), q)

        # Lower-priority restocking supports intermediate, fab, and OSAT inputs,
        # including commodities that are not themselves demanded at a sink.
        buffer_candidates = []
        for r in usable:
            dst_key = (r['dst'], r['k'])
            if dst_key in self.demand_index:
                continue
            rate = max(0.0, float(self.node_rate.get(dst_key, 0.0)))
            if rate <= 0:
                continue
            node_type = self.node_types[r['dst']] if r['dst'] < len(self.node_types) else ''
            priority = max(self.node_priority.get(dst_key, 0.0), self.max_pi * 0.15)
            if node_type in ('fab', 'osat', 'material'):
                priority = max(priority, self.max_pi * 0.25)
            duration = min(3.5, max(2.0, r['lead'] + 1.0), max(1.0, self.T - week + 1))
            target = rate * duration
            cutoff = min(self.T, week + int(np.ceil(duration)))
            need = max(0.0, target - projected_qty(dst_key, cutoff))
            if need <= 1e-9:
                continue
            score = r['cost'] / max(1.0, priority * 0.25) + 0.003 * r['lead']
            buffer_candidates.append((score, r['lead'], r['slot'], need, r))

        buffer_candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        for _, _, i, planned_need, r in buffer_candidates:
            k = r['k']
            src_key, dst_key = (r['src'], k), (r['dst'], k)
            rate = max(0.0, float(self.node_rate.get(dst_key, 0.0)))
            duration = min(3.5, max(2.0, r['lead'] + 1.0), max(1.0, self.T - week + 1))
            cutoff = min(self.T, week + int(np.ceil(duration)))
            need = min(planned_need, max(0.0, rate * duration - projected_qty(dst_key, cutoff)))
            have = max(0.0, available.get(src_key, 0.0))
            q = min(r['cap'], need, have)
            for e in r['edges']:
                q = min(q, max(0.0, float(edge_left[e])))
            for cp in r['cps']:
                q = min(q, cp_left.get((cp, self.pools[k]), 0.0))
            if q <= 1e-9 or not np.isfinite(q):
                continue
            flows[i] += q
            available[src_key] = max(0.0, have - q)
            stock_left[src_key] = max(0.0, stock_left.get(src_key, 0.0) - min(q, stock_left.get(src_key, 0.0)))
            for e in r['edges']:
                edge_left[e] = max(0.0, edge_left[e] - q)
            for cp in r['cps']:
                key = (cp, self.pools[k])
                if key in cp_left:
                    cp_left[key] = max(0.0, cp_left[key] - q)
            add_projected(dst_key, week + int(np.ceil(r['lead'])), q)

        return {'flows': flows}