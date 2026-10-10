# -2.1111482694451067
import numpy as np

class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.T = int(config['T'])
        s, l = self.s, self.l
        self.slots = s['action_slots']
        self.edges = s['edges']
        self.tail = np.asarray(self.edges['tail'], dtype=int)
        self.head = np.asarray(self.edges['head'], dtype=int)
        self.v = np.asarray(s['commodities']['v'], dtype=float)
        self.pool = s['commodities'].get('pool', [])
        self.cp_rows = {int(node): i for i, node in enumerate(l.get('chokepoints', []))}
        self.stock_rows = {tuple(map(int, x)): i for i, x in enumerate(l.get('stock_slots', []))}
        self.supply_rows = {tuple(map(int, x)): i for i, x in enumerate(l.get('supply_slots', []))}
        self.demand_rows = {tuple(map(int, x)): i for i, x in enumerate(l.get('demands', []))}
        self.routes = []
        self.out_rate = {}
        for i, (edge, k, lane) in enumerate(zip(self.slots['edge'], self.slots['k'], self.slots['lane'])):
            edge, k = int(edge), int(k)
            es = [edge] if lane is None or int(lane) < 0 else list(map(int, s['lanes']['edges'][int(lane)]))
            src, dst = int(self.tail[es[0]]), int(self.head[es[-1]])
            caps = [float(self.edges['u0'][e]) for e in es if self.edges['u0'][e] is not None]
            nominal = min(caps) if caps else 0.0
            self.routes.append((src, dst, k, es, nominal))
            self.out_rate[(src, k)] = self.out_rate.get((src, k), 0.0) + nominal
        self.lane_dest = {}
        for lane, es in enumerate(s['lanes']['edges']):
            if es:
                for e in es:
                    self.lane_dest[(int(e), lane)] = int(self.head[int(es[-1])])

    def act(self, observation):
        o = observation
        week = int(o['week'][0])
        n = len(self.routes)
        flows = np.zeros(n, dtype=float)
        stock = {key: max(0.0, float(o['stock.qty'][row])) for key, row in self.stock_rows.items()}
        available = dict(stock)
        for key, row in self.supply_rows.items():
            available[key] = available.get(key, 0.0) + max(0.0, float(o['graph_now.supply.avail'][row]))

        # Count pipeline only when it is due soon enough to cover the demand window.
        committed = {}
        pq = o.get('pipeline.qty', np.zeros(0))
        pvalid = o.get('pipeline.qty.observed', np.ones_like(pq, dtype=np.int8))
        for j in np.flatnonzero(pvalid):
            q = float(pq[j])
            if q <= 0:
                continue
            arr = int(o['pipeline.arrival_week'][j])
            if arr > week + 5:
                continue
            edge = int(o['pipeline.edge'][j])
            k = int(o['pipeline.k'][j])
            lane_valid = o.get('pipeline.lane.observed', np.ones_like(pvalid, dtype=np.int8))
            lane = int(o['pipeline.lane'][j]) if lane_valid[j] else -1
            dst = self.lane_dest.get((edge, lane), int(self.head[edge]))
            key = (dst, k)
            committed[key] = committed.get(key, 0.0) + q
        lots = o.get('queue_lots.qty')
        if lots is not None:
            for row, keyrow in enumerate(self.l.get('lot_keys', [])):
                cp, k, lane, edge = map(int, keyrow)
                q = float(np.sum(lots[row]))
                if q > 0:
                    dst = self.lane_dest.get((edge, lane), int(self.head[edge]))
                    key = (dst, k)
                    committed[key] = committed.get(key, 0.0) + q
        wq = o.get('wip.qty', np.zeros(0))
        wvalid = o.get('wip.qty.observed', np.ones_like(wq, dtype=np.int8))
        for j in np.flatnonzero(wvalid):
            if int(o['wip.out_week'][j]) <= week + 5:
                key = (int(o['wip.node'][j]), int(o['wip.k'][j]))
                committed[key] = committed.get(key, 0.0) + float(wq[j])

        u = np.asarray(o['graph_now.u'], dtype=float)
        c = np.asarray(o['graph_now.c'], dtype=float)
        tau = np.asarray(o['graph_now.tau'], dtype=float)
        tariff = np.asarray(o['graph_now.tariff'], dtype=float)
        mask = np.asarray(o['action_mask'], dtype=bool)
        opens = np.asarray(o['graph_now.open'], dtype=float)
        edge_left = np.maximum(0.0, u.copy())
        cp_left = {}
        for node, row in self.cp_rows.items():
            # Separate pools are handled by commodity class below.
            cp_left[(node, 'tb')] = max(0.0, float(o['graph_now.kappa.tb'][row]))
            cp_left[(node, 'ct')] = max(0.0, float(o['graph_now.kappa.ct'][row]))

        choices = {}
        for i, (src, dst, k, es, nominal) in enumerate(self.routes):
            if not mask[i] or nominal <= 0:
                continue
            lead = sum(max(0.0, float(tau[e])) for e in es)
            if week + max(1, lead) > self.T:
                continue
            cap = min([nominal] + [max(0.0, float(u[e])) for e in es])
            open_factor = 1.0
            cps = []
            for e in es:
                node = int(self.head[e])
                if node in self.cp_rows:
                    row = self.cp_rows[node]
                    open_factor = min(open_factor, max(0.0, opens[row]))
                    cps.append(node)
            cap *= open_factor
            if cap <= 1e-9:
                continue
            freight = sum(float(c[e]) + float(tariff[e, k]) * self.v[k] for e in es)
            # Prefer low-cost and short routes, with a penalty for currently impaired chokepoints.
            score = freight + 0.015 * self.v[k] * lead + (1.0 - open_factor) * 0.12 * self.v[k]
            choices.setdefault((dst, k), []).append((score, i, lead, cap, cps))

        needs = []
        horizon_left = max(0, self.T - week + 1)
        for key, routes in choices.items():
            dst, k = key
            lead = min(r[2] for r in routes)
            horizon = min(horizon_left, max(2.0, lead + 2.0))
            if key in self.demand_rows:
                row = self.demand_rows[key]
                forecast = np.asarray(o['demand_forecast.qty'][row], dtype=float)
                valid = o.get('demand_forecast.qty.observed')
                vals = forecast[np.asarray(valid[row], dtype=bool)] if valid is not None else forecast
                rate = max(0.0, float(np.mean(vals))) if len(vals) else 0.0
                target = rate * horizon + max(0.0, float(o['backlog.qty'][row]))
                priority = 3.0
            else:
                rate = self.out_rate.get(key, 0.0)
                target = rate * min(horizon, 2.0)
                priority = 0.8
            have = stock.get(key, 0.0) + committed.get(key, 0.0)
            need = max(0.0, target - have)
            if need > 1e-9:
                urgency = priority * need / max(rate, 1e-6)
                needs.append((-urgency, key, need))

        for _, (dst, k), need in sorted(needs):
            for score, i, lead, cap, cps in sorted(choices[(dst, k)]):
                if need <= 1e-9:
                    break
                src, _, commodity, es, nominal = self.routes[i]
                src_key = (src, commodity)
                qty = min(need, cap, available.get(src_key, 0.0))
                qty = min(qty, *(edge_left[e] for e in es))
                pool = self.pool[commodity] if commodity < len(self.pool) else 'ct'
                for node in cps:
                    qty = min(qty, cp_left.get((node, pool), 0.0))
                if qty <= 1e-9:
                    continue
                flows[i] = qty
                available[src_key] = available.get(src_key, 0.0) - qty
                need -= qty
                for e in es:
                    edge_left[e] = max(0.0, edge_left[e] - qty)
                for node in cps:
                    key = (node, pool)
                    cp_left[key] = max(0.0, cp_left.get(key, 0.0) - qty)
        return {'flows': flows}
