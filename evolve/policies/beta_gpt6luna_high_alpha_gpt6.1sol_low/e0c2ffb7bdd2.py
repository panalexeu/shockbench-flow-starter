# 0.49604062845962676
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
        self.types = s['nodes']['type']
        self.pool = s['commodities'].get('pool', ['tb'] * len(self.values))
        self.stock_slots = [tuple(map(int, p)) for p in l['stock_slots']]
        self.supply_slots = [tuple(map(int, p)) for p in l['supply_slots']]
        self.demands = [tuple(map(int, p)) for p in l['demands']]
        self.demand_index = {p: i for i, p in enumerate(self.demands)}
        self.cp_index = {int(n): i for i, n in enumerate(l['chokepoints'])}
        self.stock_index = {p: i for i, p in enumerate(self.stock_slots)}

        self.routes = []
        slots = s['action_slots']
        for i, (edge, k, lane) in enumerate(zip(slots['edge'], slots['k'], slots['lane'])):
            edge, k = int(edge), int(k)
            lane_id = -1 if lane is None else int(lane)
            if lane_id < 0:
                route_edges = [edge]
                cps = []
            else:
                route_edges = [int(x) for x in s['lanes']['edges'][lane_id]]
                cps = [int(x) for x in s['lanes']['chokepoints'][lane_id]]
            src = int(self.tail[route_edges[0]])
            dst = int(self.head[route_edges[-1]])
            self.routes.append({
                'slot': i, 'edge': edge, 'k': k, 'lane': lane_id,
                'edges': route_edges, 'cps': cps, 'src': src, 'dst': dst,
            })

        # Estimate ordinary node throughput without counting explicit alternates twice.
        grouped = {}
        for r in self.routes:
            key = (r['src'], r['k'], r['dst'])
            cap = min((self.u0[e] for e in r['edges']), default=0.0)
            if r['lane'] < 0:
                alt = edges['alt_of'][r['edge']]
            else:
                alt = s['lanes']['alt_of'][r['lane']]
            grouped.setdefault(key, []).append((alt, cap))

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

        self.pi = np.asarray(s['sinks'].get('pi', [1.0] * len(self.demands)), dtype=float)
        self.max_pi = max(float(np.max(self.pi)) if self.pi.size else 1.0, 1.0)

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

    def act(self, o):
        week = int(np.asarray(o['week']).reshape(-1)[0])
        nedge = len(self.tail)
        nslot = len(self.routes)
        u = self._field(o, 'graph_now.u', self.u0).astype(float)
        c = self._field(o, 'graph_now.c', self.c0).astype(float)
        tau = self._field(o, 'graph_now.tau', self.tau0).astype(float)
        tariff = self._field(
            o, 'graph_now.tariff', np.zeros((nedge, len(self.values)))
        ).astype(float)
        opened = self._field(o, 'graph_now.open', np.ones(len(self.cp_index))).astype(float)
        k_tb = self._field(o, 'graph_now.kappa.tb', np.full(len(self.cp_index), np.inf)).astype(float)
        k_ct = self._field(o, 'graph_now.kappa.ct', np.full(len(self.cp_index), np.inf)).astype(float)
        mask = np.asarray(o.get('action_mask', np.ones(nslot)), dtype=float).reshape(-1)

        # Current stock and supply are dispatchable now; projections also track
        # future arrivals so that they are not needlessly replaced.
        projected = {}
        available = {}
        stock = self._field(o, 'stock.qty', np.zeros(len(self.stock_slots))).astype(float)
        for i, p in enumerate(self.stock_slots):
            q = max(0.0, float(stock[i])) if i < len(stock) else 0.0
            available[p] = q
            projected.setdefault(p, []).append((week, q))
        supply = self._field(o, 'graph_now.supply.avail', np.zeros(len(self.supply_slots))).astype(float)
        for i, p in enumerate(self.supply_slots):
            q = max(0.0, float(supply[i])) if i < len(supply) else 0.0
            available[p] = available.get(p, 0.0) + q

        pqty = np.asarray(o.get('pipeline.qty', []), dtype=float).reshape(-1)
        plive = np.asarray(o.get('pipeline.qty.observed', np.ones(pqty.size)), dtype=bool).reshape(-1)
        pedge = np.asarray(o.get('pipeline.edge', np.zeros(pqty.size)), dtype=int).reshape(-1)
        pk = np.asarray(o.get('pipeline.k', np.zeros(pqty.size)), dtype=int).reshape(-1)
        plane = np.asarray(o.get('pipeline.lane', np.full(pqty.size, -1)), dtype=int).reshape(-1)
        plane_obs = np.asarray(o.get('pipeline.lane.observed', np.ones(pqty.size)), dtype=bool).reshape(-1)
        parr = np.asarray(o.get('pipeline.arrival_week', np.full(pqty.size, week)), dtype=int).reshape(-1)
        for j in range(min(pqty.size, pedge.size, pk.size, parr.size)):
            if not plive[j] or pqty[j] <= 0 or pedge[j] < 0 or pedge[j] >= nedge:
                continue
            lane = int(plane[j]) if j < plane.size and j < plane_obs.size and plane_obs[j] else -1
            dst = int(self.head[pedge[j]])
            arrival = int(parr[j])
            if 0 <= lane < len(self.s['lanes']['edges']):
                les = [int(x) for x in self.s['lanes']['edges'][lane]]
                dst = int(self.head[les[-1]])
                try:
                    pos = les.index(int(pedge[j]))
                    arrival += int(sum(max(1.0, tau[e]) for e in les[pos + 1:]))
                except ValueError:
                    pass
            projected.setdefault((dst, int(pk[j])), []).append((max(week, arrival), float(pqty[j])))

        wqty = np.asarray(o.get('wip.qty', []), dtype=float).reshape(-1)
        wlive = np.asarray(o.get('wip.qty.observed', np.ones(wqty.size)), dtype=bool).reshape(-1)
        wnode = np.asarray(o.get('wip.node', np.zeros(wqty.size)), dtype=int).reshape(-1)
        wk = np.asarray(o.get('wip.k', np.zeros(wqty.size)), dtype=int).reshape(-1)
        wout = np.asarray(o.get('wip.out_week', np.full(wqty.size, week)), dtype=int).reshape(-1)
        for j in range(min(wqty.size, wnode.size, wk.size, wout.size)):
            if j < wlive.size and wlive[j] and wqty[j] > 0:
                projected.setdefault((int(wnode[j]), int(wk[j])), []).append(
                    (max(week, int(wout[j])), float(wqty[j]))
                )

        if 'queue_lots.qty' in o:
            lots = np.asarray(o['queue_lots.qty'], dtype=float)
            lot_keys = self.l.get('lot_keys', [])
            lot_mask = np.asarray(o.get('queue_lots.qty.observed', np.ones(lots.shape)), dtype=bool)
            for row, key in enumerate(lot_keys[:lots.shape[0]]):
                if len(key) < 4:
                    continue
                _, commodity, lane, next_edge = key
                commodity, next_edge = int(commodity), int(next_edge)
                if next_edge < 0 or next_edge >= nedge:
                    continue
                if lane is not None and int(lane) >= 0 and int(lane) < len(self.s['lanes']['edges']):
                    les = [int(x) for x in self.s['lanes']['edges'][int(lane)]]
                    dst = int(self.head[les[-1]])
                    try:
                        pos = les.index(next_edge)
                        rem = sum(max(1.0, tau[e]) for e in les[pos:])
                    except ValueError:
                        rem = max(1.0, tau[next_edge])
                else:
                    dst = int(self.head[next_edge])
                    rem = max(1.0, tau[next_edge])
                for col in range(lots.shape[1]):
                    if row >= lot_mask.shape[0] or col >= lot_mask.shape[1] or not lot_mask[row, col]:
                        continue
                    q = max(0.0, float(lots[row, col]))
                    if q:
                        cohort_week = col + 1
                        arrival = max(week, cohort_week) + int(np.ceil(rem))
                        projected.setdefault((dst, commodity), []).append((arrival, q))

        pending = {}
        if 'pending_prohibitions.edge' in o:
            pe = np.asarray(o['pending_prohibitions.edge'], dtype=int).reshape(-1)
            pkp = np.asarray(o.get('pending_prohibitions.k', np.zeros(pe.size)), dtype=int).reshape(-1)
            pw = np.asarray(o.get('pending_prohibitions.effective_week', np.full(pe.size, self.T + 1)), dtype=int).reshape(-1)
            pm = np.asarray(o.get('pending_prohibitions.edge.observed', np.ones(pe.size)), dtype=bool).reshape(-1)
            for j in range(min(pe.size, pkp.size, pw.size)):
                if j < pm.size and pm[j]:
                    key = (int(pe[j]), int(pkp[j]))
                    pending[key] = min(pending.get(key, self.T + 1), int(pw[j]))

        forecast = np.asarray(o.get('demand_forecast.qty', np.zeros((len(self.demands), 1))), dtype=float)
        backlog = np.asarray(o.get('backlog.qty', np.zeros(len(self.demands))), dtype=float).reshape(-1)

        def projected_qty(p, cutoff):
            return sum(q for arrival, q in projected.get(p, ()) if arrival <= cutoff)

        def need_for(r, lead):
            dstkey = (r['dst'], r['k'])
            if dstkey in self.demand_index:
                di = self.demand_index[dstkey]
                h = min(
                    forecast.shape[1] if forecast.ndim > 1 else 1,
                    max(1, int(np.ceil(lead)) + 2),
                    max(1, self.T - week + 1),
                )
                demand = float(backlog[di]) if di < backlog.size else 0.0
                if forecast.ndim > 1 and di < forecast.shape[0]:
                    demand += float(np.sum(forecast[di, :h]))
                cutoff = week + max(0, h - 1)
            else:
                rate = max(0.0, float(self.node_rate.get(dstkey, 0.0)))
                duration = min(max(2.0, lead + 2.0), max(1.0, self.T - week + 1))
                demand = rate * duration
                cutoff = week + int(np.ceil(duration))
            return max(0.0, demand - projected_qty(dstkey, cutoff))

        edge_left = np.maximum(0.0, u.copy())
        cp_left = {}
        for cp, ci in self.cp_index.items():
            for pool_name, arr in (('tb', k_tb), ('ct', k_ct)):
                cap = float(arr[ci]) if ci < len(arr) else np.inf
                cp_left[(cp, pool_name)] = max(0.0, cap * max(0.0, opened[ci]))

        candidates = []
        for i, r in enumerate(self.routes):
            if i >= mask.size or mask[i] <= 0:
                continue
            es, cps, k = r['edges'], r['cps'], r['k']
            if not es or any(e < 0 or e >= nedge for e in es):
                continue
            if any(self.cp_index.get(cp, -1) >= 0 and opened[self.cp_index[cp]] <= 0.001 for cp in cps):
                continue
            cap = min((max(0.0, float(u[e])) for e in es), default=0.0)
            lead = sum(max(1.0, float(tau[e])) for e in es)
            if cap <= 0 or week + lead > self.T + 1:
                continue
            elapsed = 0.0
            risk = False
            for e in es:
                if pending.get((e, k), self.T + 1) <= week + elapsed:
                    risk = True
                elapsed += max(1.0, float(tau[e]))
            if risk:
                continue
            cost = sum(
                float(c[e]) + float(tariff[e, k]) * float(self.values[k])
                for e in es if e < tariff.shape[0] and k < tariff.shape[1]
            )
            for cp in cps:
                ci = self.cp_index.get(cp)
                if ci is not None:
                    cap = min(cap, cp_left.get((cp, self.pool[k]), 0.0))
            if cap <= 0 or need_for(r, lead) <= 0:
                continue
            dstkey = (r['dst'], k)
            if dstkey in self.demand_index:
                di = self.demand_index[dstkey]
                priority = max(1.0, float(self.pi[di]) if di < self.pi.size else 1.0)
            else:
                priority = max(1.0, self.max_pi * 0.25)
            score = (cost + 0.002 * float(self.values[k]) * lead) / priority
            candidates.append((score, lead, i, cap, cost))

        flows = np.zeros(nslot, dtype=float)
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        for _, lead, i, planned_cap, _ in candidates:
            r = self.routes[i]
            psrc = (r['src'], r['k'])
            pdst = (r['dst'], r['k'])
            # Recompute after each dispatch: another route may already have
            # covered some or all of this destination's requirement.
            need = need_for(r, lead)
            have = max(0.0, available.get(psrc, 0.0))
            q = min(planned_cap, need, have)
            for e in r['edges']:
                q = min(q, max(0.0, edge_left[e]))
            for cp in r['cps']:
                if cp in self.cp_index:
                    q = min(q, cp_left.get((cp, self.pool[r['k']]), 0.0))
            if q <= 0:
                continue
            flows[i] = q
            available[psrc] = max(0.0, have - q)
            for e in r['edges']:
                edge_left[e] = max(0.0, edge_left[e] - q)
            for cp in r['cps']:
                if cp in self.cp_index:
                    key = (cp, self.pool[r['k']])
                    cp_left[key] = max(0.0, cp_left.get(key, 0.0) - q)
            projected.setdefault(pdst, []).append((week + int(np.ceil(lead)), q))

        return {'flows': flows}
