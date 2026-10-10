# 0.2765283561232398
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
        self.u0 = np.asarray([0.0 if x is None else float(x) for x in e['u0']], dtype=float)
        self.c0 = np.asarray([0.0 if x is None else float(x) for x in e['c0']], dtype=float)
        self.tau0 = np.asarray([1.0 if x is None else float(x) for x in e['tau0']], dtype=float)
        self.values = np.asarray(s['commodities']['v'], dtype=float)
        self.pools = s['commodities'].get('pool', ['tb'] * len(self.values))
        self.stocks = [tuple(map(int, p)) for p in l['stock_slots']]
        self.supplies = [tuple(map(int, p)) for p in l['supply_slots']]
        self.demands = [tuple(map(int, p)) for p in l['demands']]
        self.demand_index = {p: i for i, p in enumerate(self.demands)}
        self.cp_index = {int(n): i for i, n in enumerate(l['chokepoints'])}
        self.nslots = len(s['action_slots']['edge'])
        self.node_types = s.get('nodes', {}).get('type', [])
        self.node_regions = s.get('nodes', {}).get('region', [])
        self.memory = {}

        sink_info = {}
        sinks = s.get('sinks', {})
        for n, k, pi, carry in zip(
                sinks.get('node', []), sinks.get('k', []),
                sinks.get('pi', []), sinks.get('backlog', [])):
            sink_info[(int(n), int(k))] = (max(0.01, float(pi)), bool(carry))
        self.sink_pi = np.asarray([sink_info.get(p, (1.0, False))[0] for p in self.demands])
        self.sink_carry = np.asarray([sink_info.get(p, (1.0, False))[1] for p in self.demands], dtype=bool)
        self.max_pi = max(1.0, float(np.max(self.sink_pi)) if self.sink_pi.size else 1.0)

        self.routes = []
        slots = s['action_slots']
        for i, (edge, k, lane) in enumerate(zip(slots['edge'], slots['k'], slots['lane'])):
            edge, k = int(edge), int(k)
            lane = -1 if lane is None else int(lane)
            es = [edge] if lane < 0 else [int(x) for x in s['lanes']['edges'][lane]]
            cps = [] if lane < 0 else [int(x) for x in s['lanes']['chokepoints'][lane]]
            if not es:
                continue
            self.routes.append({
                'slot': i, 'k': k, 'lane': lane, 'edges': es, 'cps': cps,
                'src': int(self.tail[es[0]]), 'dst': int(self.head[es[-1]])
            })

        # Estimate throughput at intermediate nodes, without counting marked
        # alternate routes as independent normal capacity.
        grouped = {}
        for r in self.routes:
            cap = min((self.u0[x] for x in r['edges']), default=0.0)
            edge = int(slots['edge'][r['slot']])
            if r['lane'] < 0:
                alt = e.get('alt_of', [None] * len(self.tail))[edge]
            else:
                alt = s['lanes'].get('alt_of', [None] * len(s['lanes']['edges']))[r['lane']]
            grouped.setdefault((r['src'], r['k'], r['dst']), []).append((alt, cap))
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

    def field(self, o, name, default, remember=False):
        default = np.asarray(default)
        fallback = self.memory.get(name, default) if remember else default
        if name not in o:
            return np.array(fallback, copy=True)
        x = np.asarray(o[name])
        m = o.get(name + '.observed')
        if m is not None and np.shape(m) == x.shape:
            x = np.where(m, x, fallback)
        else:
            x = x.copy()
        if remember:
            self.memory[name] = x.copy()
        return x

    def act(self, o):
        week = int(np.asarray(o['week']).reshape(-1)[0])
        ne, nk = len(self.tail), len(self.values)
        u = self.field(o, 'graph_now.u', self.u0, True).astype(float)
        c = self.field(o, 'graph_now.c', self.c0, True).astype(float)
        tau = np.maximum(1.0, self.field(o, 'graph_now.tau', self.tau0, True).astype(float))
        tariff = self.field(o, 'graph_now.tariff', np.zeros((ne, nk)), True).astype(float)
        opened = self.field(o, 'graph_now.open', np.ones(len(self.cp_index)), True).astype(float)
        mask = np.asarray(o.get('action_mask', np.ones(self.nslots)), dtype=bool).reshape(-1)
        stock_qty = self.field(o, 'stock.qty', np.zeros(len(self.stocks))).reshape(-1)
        supply_qty = self.field(o, 'graph_now.supply.avail', np.zeros(len(self.supplies)), True).reshape(-1)
        forecast = np.asarray(o.get('demand_forecast.qty', np.zeros((len(self.demands), 1))), dtype=float)
        backlog = np.asarray(o.get('backlog.qty', np.zeros(len(self.demands))), dtype=float).reshape(-1)

        stock = {}
        available = {}
        for i, p in enumerate(self.stocks):
            q = max(0.0, float(stock_qty[i])) if i < stock_qty.size else 0.0
            stock[p] = q
            available[p] = q
        for i, p in enumerate(self.supplies):
            q = max(0.0, float(supply_qty[i])) if i < supply_qty.size else 0.0
            available[p] = available.get(p, 0.0) + max(0.0, q)

        # Record known pipeline, WIP, and chokepoint lots as dated arrivals.
        arrivals = {}
        def add_arrival(pair, when, qty):
            if qty > 0 and when <= self.T + 20:
                arrivals.setdefault(pair, []).append((int(when), float(qty)))

        lanes = self.s['lanes']['edges']
        pq = np.asarray(o.get('pipeline.qty', [])).reshape(-1)
        plive = np.asarray(o.get('pipeline.qty.observed', np.ones(pq.size)), dtype=bool).reshape(-1)
        pe = np.asarray(o.get('pipeline.edge', np.zeros(pq.size)), dtype=int).reshape(-1)
        pk = np.asarray(o.get('pipeline.k', np.zeros(pq.size)), dtype=int).reshape(-1)
        plane = np.asarray(o.get('pipeline.lane', np.full(pq.size, -1)), dtype=int).reshape(-1)
        plane_live = np.asarray(o.get('pipeline.lane.observed', np.ones(pq.size)), dtype=bool).reshape(-1)
        parr = np.asarray(o.get('pipeline.arrival_week', np.full(pq.size, week)), dtype=int).reshape(-1)
        for j in range(min(pq.size, pe.size, pk.size, parr.size)):
            if j >= plive.size or not plive[j] or pq[j] <= 0 or not 0 <= pe[j] < ne:
                continue
            lane = int(plane[j]) if j < plane.size and j < plane_live.size and plane_live[j] else -1
            dst = int(self.head[pe[j]])
            arrival = int(parr[j])
            if 0 <= lane < len(lanes):
                les = [int(x) for x in lanes[lane]]
                dst = int(self.head[les[-1]])
                if int(pe[j]) in les:
                    pos = les.index(int(pe[j]))
                    arrival += int(np.ceil(sum(tau[x] for x in les[pos + 1:])))
            add_arrival((dst, int(pk[j])), max(week, arrival), pq[j])

        wq = np.asarray(o.get('wip.qty', [])).reshape(-1)
        wlive = np.asarray(o.get('wip.qty.observed', np.ones(wq.size)), dtype=bool).reshape(-1)
        wn = np.asarray(o.get('wip.node', np.zeros(wq.size)), dtype=int).reshape(-1)
        wk = np.asarray(o.get('wip.k', np.zeros(wq.size)), dtype=int).reshape(-1)
        wo = np.asarray(o.get('wip.out_week', np.full(wq.size, week)), dtype=int).reshape(-1)
        for j in range(min(wq.size, wn.size, wk.size, wo.size)):
            if j < wlive.size and wlive[j] and wq[j] > 0:
                add_arrival((int(wn[j]), int(wk[j])), max(week, int(wo[j])), wq[j])

        closure_end = {}
        ce = np.asarray(o.get('closure_end.chokepoint', [])).reshape(-1)
        cem = np.asarray(o.get('closure_end.chokepoint.observed', np.zeros(ce.size)), dtype=bool).reshape(-1)
        ew = np.asarray(o.get('closure_end.end_week', np.zeros(ce.size))).reshape(-1)
        ewm = np.asarray(o.get('closure_end.end_week.observed', np.zeros(ce.size)), dtype=bool).reshape(-1)
        for j in range(min(ce.size, cem.size, ew.size, ewm.size)):
            if cem[j] and ewm[j]:
                closure_end[int(ce[j])] = max(closure_end.get(int(ce[j]), week), int(ew[j]) + 1)
        lots = np.asarray(o.get('queue_lots.qty', np.zeros((0, 0))), dtype=float)
        lot_live = np.asarray(o.get('queue_lots.qty.observed', np.ones(lots.shape)), dtype=bool)
        for row, key in enumerate(self.l.get('lot_keys', [])[:lots.shape[0]]):
            if len(key) < 4 or row >= lot_live.shape[0]:
                continue
            cp, k, lane, next_edge = key
            cp, k, next_edge = int(cp), int(k), int(next_edge)
            if not 0 <= next_edge < ne:
                continue
            q = float(np.sum(np.where(lot_live[row], lots[row], 0.0)))
            if q <= 0:
                continue
            lane = -1 if lane is None else int(lane)
            release = week
            ci = self.cp_index.get(cp)
            if ci is not None and ci < opened.size and opened[ci] <= 0.001:
                release = closure_end.get(cp, week + 4)
            if 0 <= lane < len(lanes):
                les = [int(x) for x in lanes[lane]]
                dst = int(self.head[les[-1]])
                rem = sum(tau[x] for x in les[les.index(next_edge):]) if next_edge in les else tau[next_edge]
            else:
                dst, rem = int(self.head[next_edge]), tau[next_edge]
            add_arrival((dst, k), int(release + np.ceil(rem)), q)

        # Read pending prohibitions only where their padded rows are observed.
        pending = {}
        p_edge = np.asarray(o.get('pending_prohibitions.edge', [])).reshape(-1)
        p_k = np.asarray(o.get('pending_prohibitions.k', np.zeros(p_edge.size))).reshape(-1)
        p_week = np.asarray(o.get('pending_prohibitions.effective_week', np.full(p_edge.size, self.T + 1))).reshape(-1)
        p_live = np.asarray(o.get('pending_prohibitions.edge.observed', np.zeros(p_edge.size)), dtype=bool).reshape(-1)
        for j in range(min(p_edge.size, p_k.size, p_week.size, p_live.size)):
            if p_live[j]:
                key = (int(p_edge[j]), int(p_k[j]))
                pending[key] = min(pending.get(key, self.T + 1), int(p_week[j]))

        warning = self.field(o, 'warning.score', np.zeros(len(self.l.get('warning_units', [])))).reshape(-1)
        warning_map = {}
        for i, key in enumerate(self.l.get('warning_units', [])):
            if i < warning.size:
                warning_map[tuple(key)] = float(np.clip(warning[i], 0.0, 1.0))

        def route_warning(r):
            vals = [warning_map.get(('chokepoint', cp), 0.0) for cp in r['cps']]
            for edge in r['edges']:
                for node in (int(self.tail[edge]), int(self.head[edge])):
                    if node < len(self.node_regions):
                        vals.append(warning_map.get(('region', int(self.node_regions[node])), 0.0))
            return max(vals, default=0.0)

        # Build usable route arcs and current shared capacity budgets.
        edge_left = np.maximum(0.0, u.copy())
        cp_left = {}
        for pool in ('tb', 'ct'):
            caps = self.field(o, 'graph_now.kappa.' + pool, np.full(len(self.cp_index), np.inf), True).astype(float)
            for cp, ci in self.cp_index.items():
                cap = caps[ci] if ci < caps.size else np.inf
                op = max(0.0, float(opened[ci])) if ci < opened.size else 1.0
                cp_left[(cp, pool)] = max(0.0, float(cap) * op)

        adj = {}
        usable = {}
        for r in self.routes:
            i, k, es, cps = r['slot'], r['k'], r['edges'], r['cps']
            if i >= mask.size or not mask[i] or any(e < 0 or e >= ne for e in es):
                continue
            if any(self.cp_index.get(cp) is not None and opened[self.cp_index[cp]] <= 0.001 for cp in cps):
                continue
            lead = float(sum(tau[e] for e in es))
            if week + lead > self.T + 1:
                continue
            elapsed = 0.0
            risky_pending = False
            for e in es:
                if pending.get((e, k), self.T + 1) <= week + elapsed:
                    risky_pending = True
                    break
                elapsed += float(tau[e])
            if risky_pending:
                continue
            cap = min((max(0.0, float(u[e])) for e in es), default=0.0)
            for cp in cps:
                if cp in self.cp_index:
                    cap = min(cap, cp_left.get((cp, self.pools[k]), 0.0))
            if cap <= 0:
                continue
            freight = 0.0
            for e in es:
                if e < tariff.shape[0] and k < tariff.shape[1]:
                    freight += float(c[e]) + max(0.0, float(tariff[e, k])) * float(self.values[k])
                else:
                    freight += float(c[e])
            risk = route_warning(r)
            value = float(self.values[k])
            weight = max(0.0, freight) + 0.0015 * value * lead
            weight += value * (0.08 * risk + 0.35 * risk * risk)
            weight += 0.005 * value
            rr = dict(r, lead=lead, cap=cap, weight=weight, risk=risk)
            usable[i] = rr
            adj.setdefault((r['src'], k), []).append(rr)

        # Forecast demand with a modest extrapolation beyond the published window.
        def demand_at(di, t):
            h = max(0, int(t) - week)
            if forecast.ndim != 2 or di >= forecast.shape[0] or forecast.shape[1] == 0:
                return 0.0
            row = forecast[di]
            if h < row.size:
                return max(0.0, float(row[h]))
            tail = row[-min(3, row.size):]
            return max(0.0, float(np.mean(tail))) if tail.size else 0.0

        def event_total(pair, lo, hi):
            return sum(q for t, q in arrivals.get(pair, ()) if lo <= t <= hi)

        def sink_gap(di, arrival_week, cutoff):
            pair = self.demands[di]
            cutoff = min(self.T, int(cutoff))
            arrival_week = max(week, int(arrival_week))
            if arrival_week > cutoff:
                return 0.0, 0.0
            if self.sink_carry[di]:
                total_demand = max(0.0, float(backlog[di]) if di < backlog.size else 0.0)
                total_demand += sum(demand_at(di, t) for t in range(week, cutoff + 1))
                covered = stock.get(pair, 0.0) + event_total(pair, week, cutoff)
                return max(0.0, total_demand - covered), total_demand
            # Lost-sales stock cannot serve demand before it arrives. Simulate
            # existing inventory through the candidate's arrival date first.
            inv = stock.get(pair, 0.0)
            for t in range(week, arrival_week):
                inv += event_total(pair, t, t)
                inv = max(0.0, inv - demand_at(di, t))
            inv += event_total(pair, arrival_week, arrival_week)
            future_demand = sum(demand_at(di, t) for t in range(arrival_week, cutoff + 1))
            future_arrivals = event_total(pair, arrival_week + 1, cutoff)
            return max(0.0, future_demand - inv - future_arrivals), future_demand

        # Dijkstra from every currently supplied source. Route weights combine
        # freight, time, tariffs, and a conservative penalty for warning risk.
        sources = [(p, q) for p, q in available.items() if q > 1e-9]
        paths = {}
        for (src, k), qty in sources:
            dist = {src: 0.0}
            prev = {}
            heap = [(0.0, src)]
            while heap:
                d, node = heapq.heappop(heap)
                if d > dist.get(node, float('inf')) + 1e-10:
                    continue
                for r in adj.get((node, k), ()):
                    nd = d + r['weight']
                    if nd < dist.get(r['dst'], float('inf')) - 1e-10:
                        dist[r['dst']] = nd
                        prev[r['dst']] = (node, r)
                        heapq.heappush(heap, (nd, r['dst']))
            for target, d in dist.items():
                if target == src:
                    continue
                path = []
                node = target
                seen = set()
                while node != src and node in prev and node not in seen:
                    seen.add(node)
                    parent, r = prev[node]
                    path.append(r)
                    node = parent
                if node == src and path:
                    path.reverse()
                    paths[(src, k, target)] = (path, d)

        # Identify intermediate production/utility nodes that need a small
        # buffer even when the same commodity does not continue to a sink.
        targets_by_k = {}
        for di, (node, k) in enumerate(self.demands):
            targets_by_k.setdefault(k, []).append(('sink', int(node), di))
        internal_types = {'grid', 'fab', 'osat', 'material'}
        all_k = set(k for _, k in available) | set(r['k'] for r in self.routes)
        for k in all_k:
            graph = {}
            for r in self.routes:
                if r['k'] == k:
                    graph.setdefault(r['src'], []).append(r['dst'])
            sink_nodes = [node for node, kk in self.demands if kk == k]
            for node, typ in enumerate(self.node_types):
                if typ not in internal_types or node in sink_nodes:
                    continue
                seen = {node}
                stack = [node]
                while stack:
                    x = stack.pop()
                    for y in graph.get(x, ()):
                        if y not in seen:
                            seen.add(y)
                            stack.append(y)
                if not any(sink in seen for sink in sink_nodes):
                    targets_by_k.setdefault(k, []).append(('buffer', node, -1))

        candidates = []
        for (src, k), qty in sources:
            for kind, target, di in targets_by_k.get(k, ()):
                found = paths.get((src, k, target))
                if found is None:
                    continue
                path, route_cost = found
                if not path:
                    continue
                physical_lead = sum(r['lead'] for r in path)
                effective_lead = physical_lead + max(0, len(path) - 1)
                arrival = week + int(np.ceil(effective_lead))
                if arrival > self.T:
                    continue
                risk = max((r['risk'] for r in path), default=0.0)
                if kind == 'sink':
                    buffer = 2 + int(round(3.0 * risk))
                    cutoff = min(self.T, arrival + buffer)
                    gap, target_demand = sink_gap(di, arrival, cutoff)
                    if gap <= 1e-9:
                        continue
                    pi = max(0.01, float(self.sink_pi[di]))
                    urgency = 0.35 + 0.65 * min(1.0, gap / max(target_demand, 1e-9))
                    score = route_cost / (pi * urgency)
                else:
                    pair = (target, k)
                    typ = self.node_types[target] if target < len(self.node_types) else ''
                    rate = max(0.0, float(self.node_rate.get(pair, 0.0)))
                    duration = min(3.0 + 2.0 * risk, max(1.5, effective_lead + 1.0))
                    cutoff = min(self.T, arrival + int(np.ceil(duration)))
                    goal = rate * duration
                    gap = max(0.0, goal - stock.get(pair, 0.0) - event_total(pair, week, cutoff))
                    if gap <= 1e-9:
                        continue
                    importance = 0.70 if typ in ('grid', 'fab', 'osat') else 0.55
                    priority = max(1.0, self.max_pi * importance)
                    score = route_cost / priority
                candidates.append((score, arrival, kind, di, target, k, path, gap))

        candidates.sort(key=lambda x: (x[0], x[1], x[2], x[4]))
        flows = np.zeros(self.nslots, dtype=float)
        for _, arrival, kind, di, target, k, path, planned_gap in candidates:
            first = path[0]
            pair_src = (first['src'], k)
            pair_dst = (target, k)
            if kind == 'sink':
                risk = max((r['risk'] for r in path), default=0.0)
                cutoff = min(self.T, arrival + 2 + int(round(3.0 * risk)))
                gap, _ = sink_gap(di, arrival, cutoff)
            else:
                risk = max((r['risk'] for r in path), default=0.0)
                duration = min(3.0 + 2.0 * risk, max(1.5, (arrival - week) + 1.0))
                cutoff = min(self.T, arrival + int(np.ceil(duration)))
                goal = max(0.0, float(self.node_rate.get(pair_dst, 0.0))) * duration
                gap = max(0.0, goal - stock.get(pair_dst, 0.0) - event_total(pair_dst, week, cutoff))
            q = min(max(0.0, available.get(pair_src, 0.0)), max(0.0, gap), first['cap'])
            for e in first['edges']:
                q = min(q, max(0.0, edge_left[e]))
            for cp in first['cps']:
                if cp in self.cp_index:
                    q = min(q, cp_left.get((cp, self.pools[k]), 0.0))
            if not np.isfinite(q) or q <= 1e-9:
                continue
            flows[first['slot']] += q
            available[pair_src] = max(0.0, available.get(pair_src, 0.0) - q)
            for e in first['edges']:
                edge_left[e] = max(0.0, edge_left[e] - q)
            for cp in first['cps']:
                if cp in self.cp_index:
                    key = (cp, self.pools[k])
                    cp_left[key] = max(0.0, cp_left.get(key, 0.0) - q)
            add_arrival(pair_dst, arrival, q)

        return {'flows': flows}
