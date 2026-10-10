# -2.1122216085998775
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
        self.dem = {tuple(map(int, x)): i for i, x in enumerate(self.l.get('demands', []))}
        self.supply = {tuple(map(int, x)): i for i, x in enumerate(self.l.get('supply_slots', []))}
        self.stock_rows = {tuple(map(int, x)): i for i, x in enumerate(self.l.get('stock_slots', []))}
        self.cp = {int(node): i for i, node in enumerate(self.l.get('chokepoints', []))}
        self.v = np.asarray(self.s['commodities']['v'], dtype=float)
        self.pi = np.asarray(self.s.get('sinks', {}).get('pi', []), dtype=float)
        self.routes = []
        self.lane_dest = {}
        for lane, es0 in enumerate(self.s.get('lanes', {}).get('edges', [])):
            es = list(map(int, es0))
            if es:
                for e in es:
                    self.lane_dest[(e, lane)] = int(self.head[es[-1]])
        # Approximate normal dispatch rates to intermediate nodes using distinct
        # source/destination routes, so transfers are not needlessly stockpiled.
        by_src_dest = {}
        for i, (edge0, k0, lane0) in enumerate(zip(self.slots['edge'], self.slots['k'], self.slots['lane'])):
            edge, k = int(edge0), int(k0)
            es = [edge] if lane0 is None or int(lane0) < 0 else list(map(int, self.s['lanes']['edges'][int(lane0)]))
            if not es:
                es = [edge]
            src, dst = int(self.tail[es[0]]), int(self.head[es[-1]])
            caps = [float(self.edges['u0'][e]) for e in es if self.edges['u0'][e] is not None]
            nominal = min(caps) if caps else 0.0
            self.routes.append((src, dst, k, es, nominal))
            by_src_dest[(src, dst, k)] = max(by_src_dest.get((src, dst, k), 0.0), nominal)
        self.rate = {}
        for (src, dst, k), cap in by_src_dest.items():
            self.rate[(src, k)] = self.rate.get((src, k), 0.0) + cap

    def act(self, o):
        t = int(o['week'][0])
        stock = {key: max(0.0, float(o['stock.qty'][row])) for key, row in self.stock_rows.items()}
        source_left = dict(stock)
        for key, row in self.supply.items():
            source_left[key] = source_left.get(key, 0.0) + max(0.0, float(o['graph_now.supply.avail'][row]))

        # Estimate near-term receipts at route destinations. Do not let distant
        # arrivals suppress replenishment for the current demand window.
        committed = {}
        pq = o.get('pipeline.qty', np.zeros(0))
        pm = o.get('pipeline.qty.observed', np.ones_like(pq, dtype=np.int8))
        plm = o.get('pipeline.lane.observed', np.ones_like(pm, dtype=np.int8))
        for j in np.flatnonzero(pm):
            q = max(0.0, float(pq[j]))
            if q <= 0:
                continue
            arr = int(o['pipeline.arrival_week'][j])
            edge, k = int(o['pipeline.edge'][j]), int(o['pipeline.k'][j])
            lane = int(o['pipeline.lane'][j]) if plm[j] else -1
            dst = self.lane_dest.get((edge, lane), int(self.head[edge]))
            key = (dst, k)
            # Capped later per destination after its replenishment window is known.
            committed.setdefault(key, []).append((arr, q))
        lots = o.get('queue_lots.qty')
        if lots is not None:
            for r, keyrow in enumerate(self.l.get('lot_keys', [])):
                cp, k, lane, edge = map(int, keyrow)
                q = float(np.sum(lots[r]))
                if q > 0:
                    dst = self.lane_dest.get((edge, lane), int(self.head[edge]))
                    committed.setdefault((dst, k), []).append((t, q))
        wm = o.get('wip.qty.observed', np.zeros_like(o.get('wip.qty', []), dtype=np.int8))
        for j in np.flatnonzero(wm):
            q = max(0.0, float(o['wip.qty'][j]))
            if q > 0:
                key = (int(o['wip.node'][j]), int(o['wip.k'][j]))
                committed.setdefault(key, []).append((int(o['wip.out_week'][j]), q))

        u = np.asarray(o['graph_now.u'], dtype=float)
        c = np.asarray(o['graph_now.c'], dtype=float)
        tau = np.asarray(o['graph_now.tau'], dtype=float)
        tariff = np.asarray(o['graph_now.tariff'], dtype=float)
        mask = np.asarray(o['action_mask'], dtype=bool)
        edge_left = np.maximum(0.0, u.copy())
        candidates = {}
        for i, (src, dst, k, es, nominal) in enumerate(self.routes):
            if not mask[i] or nominal <= 0:
                continue
            lead = max(1.0, sum(max(0.0, float(tau[e])) for e in es))
            if t + lead > self.T:
                continue
            cap = min([nominal] + [max(0.0, float(u[e])) for e in es])
            openness = 1.0
            for e in es:
                row = self.cp.get(int(self.head[e]))
                if row is not None:
                    openness = min(openness, max(0.0, float(o['graph_now.open'][row])))
            cap *= openness
            if cap <= 1e-9:
                continue
            freight = sum(float(c[e]) + float(tariff[e, k]) * self.v[k] for e in es)
            score = freight + 0.02 * self.v[k] * lead + (1.0 - openness) * 0.12 * self.v[k]
            candidates.setdefault((dst, k), []).append((score, i, lead, cap))

        flows = np.zeros(self.nslot, dtype=float)
        priorities = []
        forecast = np.asarray(o['demand_forecast.qty'], dtype=float)
        fmask = o.get('demand_forecast.qty.observed')
        for key, routes in candidates.items():
            dst, k = key
            lead = min(x[2] for x in routes)
            remaining = max(0.0, self.T - t + 1.0)
            # A modest buffer covers variability without treating all future
            # supply as an immediate need.
            horizon = min(remaining, max(1.0, lead + 1.0))
            if key in self.dem:
                row = self.dem[key]
                n = min(forecast.shape[1], max(1, int(np.ceil(horizon))))
                total = 0.0
                for h in range(n):
                    weight = min(1.0, max(0.0, horizon - h))
                    if weight <= 0:
                        continue
                    if fmask is None or bool(fmask[row, h]):
                        val = max(0.0, float(forecast[row, h]))
                    else:
                        val = 0.0
                    total += weight * val
                total += max(0.0, float(o['backlog.qty'][row]))
                # Use penalty as a tie-breaker between competing sink needs.
                penalty = float(self.pi[row]) if row < len(self.pi) else float(self.v[k])
                rate = total / max(horizon, 1.0)
                have = stock.get(key, 0.0)
                arrivals = committed.get(key, [])
                have += sum(q for arr, q in arrivals if arr <= t + horizon)
                need = max(0.0, total - have)
                urgency = (1.0 + max(0.0, penalty) / max(1.0, float(self.v[k]))) * need / max(rate, 1e-9)
            else:
                rate = self.rate.get(key, 0.0)
                target = rate * min(horizon, 2.0)
                have = stock.get(key, 0.0)
                have += sum(q for arr, q in committed.get(key, []) if arr <= t + horizon)
                need = max(0.0, target - have)
                urgency = 0.25 * need / max(rate, 1e-9)
            if need > 1e-9:
                priorities.append((-urgency, key, need))

        for _, key, need in sorted(priorities):
            for score, i, lead, cap in sorted(candidates[key]):
                if need <= 1e-9:
                    break
                src, dst, k, es, nominal = self.routes[i]
                avail = source_left.get((src, k), 0.0)
                qty = min(need, cap, avail)
                for e in es:
                    qty = min(qty, edge_left[e])
                if qty <= 1e-9:
                    continue
                flows[i] += qty
                source_left[(src, k)] = avail - qty
                need -= qty
                for e in es:
                    edge_left[e] = max(0.0, edge_left[e] - qty)
        return {'flows': flows}