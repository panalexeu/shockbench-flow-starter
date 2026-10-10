# -2.108665345318672
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
        self.cp_row = {int(n): i for i, n in enumerate(self.l.get('chokepoints', []))}
        self.stock_row = {tuple(map(int, x)): i for i, x in enumerate(self.l.get('stock_slots', []))}
        self.supply_row = {tuple(map(int, x)): i for i, x in enumerate(self.l.get('supply_slots', []))}
        self.demand_row = {tuple(map(int, x)): i for i, x in enumerate(self.l.get('demands', []))}
        sinks = self.s.get('sinks', {})
        self.penalty = np.asarray(sinks.get('pi', []), dtype=float)
        self.lane_edges = self.s.get('lanes', {}).get('edges', [])
        self.routes = []
        for i, (e0, k0, lane0) in enumerate(zip(self.slots['edge'], self.slots['k'], self.slots['lane'])):
            e, k = int(e0), int(k0)
            es = [e] if lane0 is None or int(lane0) < 0 else list(map(int, self.lane_edges[int(lane0)]))
            if not es:
                es = [e]
            self.routes.append((i, int(self.tail[es[0]]), int(self.head[es[-1]]), k, es))
        self.lane_dest = {}
        for lane, edges in enumerate(self.lane_edges):
            if edges:
                dst = int(self.head[int(edges[-1])])
                for edge in edges:
                    self.lane_dest[(int(edge), lane)] = dst

    def _dest(self, edge, lane):
        return self.lane_dest.get((edge, lane), int(self.head[edge]))

    def act(self, o):
        week = int(o['week'][0])
        nnode = len(self.s['nodes']['id'])
        nk = len(self.s['commodities']['id'])
        available = {}
        stock = {}
        for key, row in self.stock_row.items():
            q = max(0.0, float(o['stock.qty'][row]))
            stock[key] = q
            available[key] = q
        for key, row in self.supply_row.items():
            available[key] = available.get(key, 0.0) + max(0.0, float(o['graph_now.supply.avail'][row]))

        # Estimate already-committed near-term supply at each destination.
        incoming = {}
        pq = o.get('pipeline.qty', np.zeros(0))
        pm = np.asarray(o.get('pipeline.qty.observed', np.ones_like(pq, dtype=np.int8)), dtype=bool)
        lm = np.asarray(o.get('pipeline.lane.observed', np.ones_like(pm, dtype=np.int8)), dtype=bool)
        arrival = o.get('pipeline.arrival_week', np.zeros_like(pq, dtype=int))
        for j in np.flatnonzero(pm):
            q = max(0.0, float(pq[j]))
            if q <= 0 or int(arrival[j]) > week + 7:
                continue
            e, k = int(o['pipeline.edge'][j]), int(o['pipeline.k'][j])
            lane = int(o['pipeline.lane'][j]) if lm[j] else -1
            key = (self._dest(e, lane), k)
            incoming[key] = incoming.get(key, 0.0) + q
        lots = o.get('queue_lots.qty')
        if lots is not None:
            for r, lotkey in enumerate(self.l.get('lot_keys', [])):
                cp, k, lane, edge = map(int, lotkey)
                q = float(np.sum(lots[r]))
                if q > 0:
                    key = (self._dest(edge, lane), k)
                    incoming[key] = incoming.get(key, 0.0) + q
        wq = o.get('wip.qty', np.zeros(0))
        wm = np.asarray(o.get('wip.qty.observed', np.ones_like(wq, dtype=np.int8)), dtype=bool)
        for j in np.flatnonzero(wm):
            if int(o['wip.out_week'][j]) <= week + 7:
                key = (int(o['wip.node'][j]), int(o['wip.k'][j]))
                incoming[key] = incoming.get(key, 0.0) + max(0.0, float(wq[j]))

        u = np.asarray(o['graph_now.u'], dtype=float)
        c = np.asarray(o['graph_now.c'], dtype=float)
        tau = np.asarray(o['graph_now.tau'], dtype=float)
        tariff = np.asarray(o['graph_now.tariff'], dtype=float)
        mask = np.asarray(o['action_mask'], dtype=bool)
        opened = np.asarray(o['graph_now.open'], dtype=float)
        edge_left = np.maximum(0.0, u.copy())
        cp_left = {}
        for node, row in self.cp_row.items():
            cp_left[(node, 'tb')] = max(0.0, float(o['graph_now.kappa.tb'][row]))
            cp_left[(node, 'ct')] = max(0.0, float(o['graph_now.kappa.ct'][row]))

        routes_by_sink = {}
        for slot, src, dst, k, es in self.routes:
            if not mask[slot]:
                continue
            lead = sum(max(0.0, float(tau[e])) for e in es)
            if week + max(1.0, lead) > self.T:
                continue
            op = 1.0
            cps = []
            for e in es:
                node = int(self.head[e])
                if node in self.cp_row:
                    row = self.cp_row[node]
                    op = min(op, max(0.0, float(opened[row])))
                    cps.append(node)
            if op <= 1e-8:
                continue
            nominal = [float(self.edges['u0'][e]) for e in es if self.edges['u0'][e] is not None]
            basecap = min(nominal) if nominal else 0.0
            cap = min([basecap] + [max(0.0, float(u[e])) for e in es]) * op
            if cap <= 1e-9:
                continue
            freight = sum(float(c[e]) + float(tariff[e, k]) * self.v[k] for e in es)
            cost = freight + 0.018 * self.v[k] * lead + (1.0 - op) * 0.12 * self.v[k]
            pool = self.pool[k] if k < len(self.pool) else 'ct'
            routes_by_sink.setdefault((dst, k), []).append((cost, lead, slot, src, dst, k, es, cap, cps, pool))

        forecasts = np.asarray(o['demand_forecast.qty'], dtype=float)
        fmask = o.get('demand_forecast.qty.observed')
        needs = []
        for key, row in self.demand_row.items():
            options = routes_by_sink.get(key, [])
            if not options:
                continue
            options.sort(key=lambda x: x[0] + 0.015 * self.v[key[1]] * x[1])
            best = options[0]
            vals = forecasts[row]
            if fmask is not None:
                vals = vals[np.asarray(fmask[row], dtype=bool)]
            rate = max(0.0, float(np.mean(vals))) if len(vals) else 0.0
            lead = best[1]
            horizon = min(max(0, self.T - week + 1), max(1.0, lead + 1.5))
            target = rate * horizon + max(0.0, float(o['backlog.qty'][row]))
            have = stock.get(key, 0.0) + incoming.get(key, 0.0)
            need = max(0.0, target - have)
            if need <= 1e-9:
                continue
            pi = float(self.penalty[row]) if row < len(self.penalty) else float(np.max(self.v))
            urgency = pi * (need + float(o['backlog.qty'][row])) / max(1e-9, rate * horizon)
            needs.append((-urgency, -pi, key, need, options))

        flows = np.zeros(self.nslot, dtype=float)
        for _, _, key, need0, options in sorted(needs):
            need = need0
            for cost, lead, slot, src, dst, k, es, cap, cps, pool in options:
                if need <= 1e-9:
                    break
                qty = min(need, cap, available.get((src, k), 0.0))
                for e in es:
                    qty = min(qty, edge_left[e])
                for node in cps:
                    qty = min(qty, cp_left.get((node, pool), 0.0))
                if qty <= 1e-9:
                    continue
                flows[slot] += qty
                available[(src, k)] = max(0.0, available.get((src, k), 0.0) - qty)
                need -= qty
                for e in es:
                    edge_left[e] = max(0.0, edge_left[e] - qty)
                for node in cps:
                    cp_left[(node, pool)] = max(0.0, cp_left.get((node, pool), 0.0) - qty)
        return {'flows': flows}
