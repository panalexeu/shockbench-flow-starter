# -2.107878143996054
import numpy as np

class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.T = int(config['T'])
        self.edges = self.s['edges']
        self.tail = np.asarray(self.edges['tail'], dtype=int)
        self.head = np.asarray(self.edges['head'], dtype=int)
        self.slots = self.s['action_slots']
        self.nslot = len(self.slots['edge'])
        self.v = np.asarray(self.s['commodities']['v'], dtype=float)
        self.pool = self.s['commodities'].get('pool', [])
        self.stock_rows = {tuple(map(int, x)): i for i, x in enumerate(self.l.get('stock_slots', []))}
        self.supply_rows = {tuple(map(int, x)): i for i, x in enumerate(self.l.get('supply_slots', []))}
        self.demand_rows = {tuple(map(int, x)): i for i, x in enumerate(self.l.get('demands', []))}
        self.cp_rows = {int(n): i for i, n in enumerate(self.l.get('chokepoints', []))}
        self.penalty = np.asarray(self.s.get('sinks', {}).get('pi', []), dtype=float)
        self.backlog_flags = self.s.get('sinks', {}).get('backlog', [])
        self.lane_edges = [[int(e) for e in es] for es in self.s.get('lanes', {}).get('edges', [])]
        self.routes = []
        self.lane_dest = {}
        for lane, es in enumerate(self.lane_edges):
            if es:
                dst = int(self.head[es[-1]])
                for e in es:
                    self.lane_dest[(e, lane)] = dst
        # Route graph, retaining a slot index so parallel alternatives remain usable.
        for i, (edge0, k0, lane0) in enumerate(zip(self.slots['edge'], self.slots['k'], self.slots['lane'])):
            edge, k = int(edge0), int(k0)
            es = [edge] if lane0 is None or int(lane0) < 0 else self.lane_edges[int(lane0)]
            if not es:
                continue
            src, dst = int(self.tail[es[0]]), int(self.head[es[-1]])
            caps = [float(self.edges['u0'][e]) for e in es if self.edges['u0'][e] is not None]
            nominal = min(caps) if caps else 0.0
            self.routes.append({'slot': i, 'src': src, 'dst': dst, 'k': k, 'es': es, 'nom': nominal})
        # Approximate normal weekly draw at intermediate source/stock nodes.
        self.nominal_out = {}
        unique = {}
        for r in self.routes:
            key = (r['src'], r['dst'], r['k'])
            unique[key] = max(unique.get(key, 0.0), r['nom'])
        for (src, dst, k), cap in unique.items():
            self.nominal_out[(src, k)] = self.nominal_out.get((src, k), 0.0) + cap
        self.n_nodes = len(self.s['nodes']['id'])

    def act(self, o):
        t = int(o['week'][0])
        ncommod = len(self.s['commodities']['id'])
        stock = np.zeros((self.n_nodes, ncommod), dtype=float)
        for key, row in self.stock_rows.items():
            stock[key] = max(0.0, float(o['stock.qty'][row]))
        available = stock.copy()
        for key, row in self.supply_rows.items():
            available[key] += max(0.0, float(o['graph_now.supply.avail'][row]))

        # Demand targets are kept local to sinks; forecast horizon includes delivery lead.
        forecast = np.asarray(o['demand_forecast.qty'], dtype=float)
        fmask = o.get('demand_forecast.qty.observed')
        backlog = np.asarray(o.get('backlog.qty', np.zeros(len(self.demand_rows))), dtype=float)
        demand_rate = {}
        target = {}
        for key, row in self.demand_rows.items():
            vals = forecast[row]
            if fmask is not None:
                vals = vals[np.asarray(fmask[row], dtype=bool)]
            rate = max(0.0, float(np.mean(vals))) if len(vals) else 0.0
            demand_rate[key] = rate
            target[key] = max(0.0, float(backlog[row]))

        # Pipeline arrivals and queued lots are assigned to their true route destination.
        committed = {}
        pq = o.get('pipeline.qty', np.zeros(0))
        pm = o.get('pipeline.qty.observed', np.ones_like(pq, dtype=np.int8))
        plm = o.get('pipeline.lane.observed', np.ones_like(pm, dtype=np.int8))
        arrivals = {}
        for j in np.flatnonzero(pm):
            q = max(0.0, float(pq[j]))
            if q <= 0:
                continue
            e, k = int(o['pipeline.edge'][j]), int(o['pipeline.k'][j])
            lane = int(o['pipeline.lane'][j]) if plm[j] else -1
            dst = self.lane_dest.get((e, lane), int(self.head[e]))
            arr = int(o['pipeline.arrival_week'][j])
            arrivals.setdefault((dst, k), []).append((arr, q))
        lots = o.get('queue_lots.qty')
        if lots is not None:
            for row, keyrow in enumerate(self.l.get('lot_keys', [])):
                cp, k, lane, edge = map(int, keyrow)
                q = float(np.sum(lots[row]))
                if q > 0:
                    dst = self.lane_dest.get((edge, lane), int(self.head[edge]))
                    arrivals.setdefault((dst, k), []).append((t + 1, q))
        wq = o.get('wip.qty', np.zeros(0))
        wm = o.get('wip.qty.observed', np.ones_like(wq, dtype=np.int8))
        for j in np.flatnonzero(wm):
            q = max(0.0, float(wq[j]))
            if q > 0:
                key = (int(o['wip.node'][j]), int(o['wip.k'][j]))
                arrivals.setdefault(key, []).append((int(o['wip.out_week'][j]), q))

        # Forecast near-term demand over each sink's delivery window. Only arrivals
        # that are expected by that window reduce the amount to route.
        for key, row in self.demand_rows.items():
            opts = [r for r in self.routes if (r['dst'], r['k']) == key]
            if not opts:
                continue
            tau = np.asarray(o['graph_now.tau'], dtype=float)
            lead = min(max(1.0, sum(max(0.0, float(tau[e])) for e in r['es'])) for r in opts)
            h = min(max(0, self.T - t + 1), lead + 2.0, forecast.shape[1])
            n = int(np.ceil(h))
            amount = 0.0
            for z in range(n):
                if fmask is None or bool(fmask[row, z]):
                    amount += max(0.0, float(forecast[row, z])) * min(1.0, max(0.0, h-z))
            target[key] += amount
            target[key] = max(0.0, target[key] - sum(q for arr, q in arrivals.get(key, []) if arr <= t + h))

        u = np.maximum(0.0, np.asarray(o['graph_now.u'], dtype=float))
        tau = np.asarray(o['graph_now.tau'], dtype=float)
        cost = np.asarray(o['graph_now.c'], dtype=float)
        tariff = np.asarray(o['graph_now.tariff'], dtype=float)
        mask = np.asarray(o['action_mask'], dtype=bool)
        opened = np.asarray(o['graph_now.open'], dtype=float)
        ktb = np.asarray(o['graph_now.kappa.tb'], dtype=float)
        kct = np.asarray(o['graph_now.kappa.ct'], dtype=float)
        edge_left = u.copy()
        cp_left = {}
        for node, row in self.cp_rows.items():
            cp_left[(node, 'tb')] = max(0.0, float(ktb[row]))
            cp_left[(node, 'ct')] = max(0.0, float(kct[row]))

        # Remaining sink demand plus a small replenishment target at intermediate
        # nodes allows the policy to move goods through multi-edge paths.
        need = {}
        for key in self.stock_rows:
            node, k = key
            if key in self.demand_rows:
                need[key] = max(0.0, target.get(key, 0.0) - stock[node, k])
            else:
                rate = self.nominal_out.get(key, 0.0)
                need[key] = max(0.0, min(rate, available[node, k]) - stock[node, k])
        for key in self.demand_rows:
            node, k = key
            if key not in need:
                need[key] = max(0.0, target.get(key, 0.0) - stock[node, k])

        flows = np.zeros(self.nslot, dtype=float)
        # Repeatedly select the currently most valuable feasible shipment. A route
        # may serve a downstream sink directly or replenish a useful intermediate.
        candidates = []
        for r in self.routes:
            i, src, dst, k, es = r['slot'], r['src'], r['dst'], r['k'], r['es']
            if not mask[i] or r['nom'] <= 0 or t + max(1.0, sum(max(0.0, float(tau[e])) for e in es)) > self.T:
                continue
            op = 1.0
            cps = []
            prohibited = False
            for e in es:
                cp_row = self.cp_rows.get(int(self.head[e]))
                if cp_row is not None:
                    op = min(op, max(0.0, float(opened[cp_row])))
                    cps.append(int(self.head[e]))
            if op <= 0.01:
                continue
            cap = min([r['nom']] + [float(u[e]) for e in es]) * op
            if cap <= 1e-9:
                continue
            pool = self.pool[k] if k < len(self.pool) else 'ct'
            for node in cps:
                cap = min(cap, cp_left.get((node, pool), 0.0))
            if cap <= 1e-9:
                continue
            lead = max(1.0, sum(max(0.0, float(tau[e])) for e in es))
            freight = sum(float(cost[e]) + float(tariff[e, k]) * self.v[k] for e in es)
            route_cost = freight + 0.015 * self.v[k] * lead + (1.0-op) * 0.12 * self.v[k]
            candidates.append((route_cost, i, src, dst, k, es, cap, lead, cps, pool))

        # Allocate in urgency order, recalculating source and edge balances after
        # each assignment. Deliveries to intermediates are limited to their draw.
        for _ in range(max(1, len(candidates) * 2)):
            best = None
            for item in candidates:
                route_cost, i, src, dst, k, es, cap, lead, cps, pool = item
                residual = max(0.0, need.get((dst, k), 0.0))
                if residual <= 1e-8:
                    continue
                qty = min(residual, cap, available[src, k], flows[i] * 0 + min(edge_left[e] for e in es))
                for node in cps:
                    qty = min(qty, cp_left.get((node, pool), 0.0))
                if qty <= 1e-8:
                    continue
                rate = demand_rate.get((dst, k), self.nominal_out.get((dst, k), 0.0))
                priority = 1.0 + (self.penalty[self.demand_rows[(dst,k)]] if (dst,k) in self.demand_rows and self.demand_rows[(dst,k)] < len(self.penalty) else 0.0) / max(1.0, self.v[k])
                urgency = priority * residual / max(1.0, rate)
                value = urgency / (1.0 + route_cost / max(1.0, self.v[k]))
                if best is None or value > best[0]:
                    best = (value, item, qty)
            if best is None:
                break
            _, item, qty = best
            route_cost, i, src, dst, k, es, cap, lead, cps, pool = item
            flows[i] += qty
            available[src, k] -= qty
            need[(dst, k)] = max(0.0, need.get((dst, k), 0.0) - qty)
            for e in es:
                edge_left[e] = max(0.0, edge_left[e] - qty)
            for node in cps:
                cp_left[(node, pool)] = max(0.0, cp_left.get((node, pool), 0.0) - qty)
            # A newly supplied intermediate can feed later legs only next week;
            # don't spend it again in the same action.
        return {'flows': flows}
