# -2.112135433010746
import numpy as np

class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.T = int(config['T'])
        s, l = self.s, self.l
        self.edges = s['edges']
        self.tail = np.asarray(self.edges['tail'], dtype=int)
        self.head = np.asarray(self.edges['head'], dtype=int)
        self.v = np.asarray(s['commodities']['v'], dtype=float)
        self.pool = s['commodities'].get('pool', [])
        self.cp_rows = {int(n): i for i, n in enumerate(l.get('chokepoints', []))}
        self.stock_rows = {tuple(map(int, x)): i for i, x in enumerate(l.get('stock_slots', []))}
        self.supply_rows = {tuple(map(int, x)): i for i, x in enumerate(l.get('supply_slots', []))}
        self.dem_rows = {tuple(map(int, x)): i for i, x in enumerate(l.get('demands', []))}
        self.routes = []
        slots = s['action_slots']
        for edge, k, lane in zip(slots['edge'], slots['k'], slots['lane']):
            edge, k = int(edge), int(k)
            es = [edge] if lane is None or int(lane) < 0 else list(map(int, s['lanes']['edges'][int(lane)]))
            src = int(self.tail[es[0]])
            dst = int(self.head[es[-1]])
            caps = [float(self.edges['u0'][e]) for e in es if self.edges['u0'][e] is not None]
            nominal = min(caps) if caps else 0.0
            cps = []
            for e in es:
                node = int(self.head[e])
                if node in self.cp_rows and node not in cps:
                    cps.append(node)
            self.routes.append((src, dst, k, es, nominal, cps))
        self.edge_dest = {}
        for lane, es in enumerate(s['lanes']['edges']):
            if es:
                dst = int(self.head[int(es[-1])])
                for e in es:
                    self.edge_dest[(int(e), lane)] = dst
        # Nominal output-rate estimates for intermediate goods and fallback forecasts.
        self.rate = {}
        unique = {}
        for src, dst, k, es, cap, cps in self.routes:
            unique[(src, dst, k)] = max(unique.get((src, dst, k), 0.0), cap)
        for (src, dst, k), cap in unique.items():
            self.rate[(src, k)] = self.rate.get((src, k), 0.0) + cap
        for key in self.dem_rows:
            self.rate.setdefault(key, 0.0)
        self.last_rate = {}

    def act(self, o):
        t = int(o['week'][0])
        n = len(self.routes)
        flows = np.zeros(n, dtype=float)
        stock = {key: max(0.0, float(o['stock.qty'][row])) for key, row in self.stock_rows.items()}
        available = dict(stock)
        for key, row in self.supply_rows.items():
            available[key] = available.get(key, 0.0) + max(0.0, float(o['graph_now.supply.avail'][row]))

        # Pending prohibitions are checked at the estimated time of passage.
        pending = {}
        pe = o.get('pending_prohibitions.edge', np.zeros(0, dtype=int))
        pem = o.get('pending_prohibitions.edge.observed', np.ones_like(pe, dtype=np.int8))
        ew = o.get('pending_prohibitions.effective_week', np.zeros_like(pe))
        ewm = o.get('pending_prohibitions.effective_week.observed', np.ones_like(pe, dtype=np.int8))
        for j in np.flatnonzero(pem):
            if ewm[j]:
                key = (int(pe[j]), int(o['pending_prohibitions.k'][j]))
                pending[key] = min(pending.get(key, self.T + 100), int(ew[j]))

        tau = np.asarray(o['graph_now.tau'], dtype=float)
        opened = np.asarray(o['graph_now.open'], dtype=float)
        committed = {}

        def reliability(cps, arrival):
            # Current closures can delay queued/in-flight cargo; do not count it as certain stock.
            factor = 1.0
            for node in cps:
                row = self.cp_rows[node]
                op = max(0.0, float(opened[row]))
                if op < 0.05:
                    factor *= 0.45
                elif op < 0.5:
                    factor *= 0.75
            return factor

        pq = o.get('pipeline.qty', np.zeros(0))
        qmask = o.get('pipeline.qty.observed', np.ones_like(pq, dtype=np.int8))
        lmask = o.get('pipeline.lane.observed', np.ones_like(qmask, dtype=np.int8))
        for j in np.flatnonzero(qmask):
            q = max(0.0, float(pq[j]))
            if q <= 0:
                continue
            e, k = int(o['pipeline.edge'][j]), int(o['pipeline.k'][j])
            lane = int(o['pipeline.lane'][j]) if lmask[j] else -1
            dst = self.edge_dest.get((e, lane), int(self.head[e]))
            arr = int(o['pipeline.arrival_week'][j])
            if arr > t + 8:
                continue
            cps = list(map(int, self.s['lanes']['chokepoints'][lane])) if lane >= 0 else []
            # A shipment arriving later than the useful forecast window contributes little to today's plan.
            discount = reliability(cps, arr) * (1.0 if arr <= t + 4 else 0.65)
            key = (dst, k)
            committed[key] = committed.get(key, 0.0) + q * discount

        lots = o.get('queue_lots.qty')
        if lots is not None:
            for row, keyrow in enumerate(self.l.get('lot_keys', [])):
                cp, k, lane, edge = map(int, keyrow)
                q = max(0.0, float(np.sum(lots[row])))
                if q <= 0:
                    continue
                dst = self.edge_dest.get((edge, lane), int(self.head[edge]))
                op = float(opened[self.cp_rows[cp]]) if cp in self.cp_rows else 1.0
                discount = 0.4 if op < 0.05 else (0.7 if op < 0.5 else 0.95)
                key = (dst, k)
                committed[key] = committed.get(key, 0.0) + q * discount

        wq = o.get('wip.qty', np.zeros(0))
        wmask = o.get('wip.qty.observed', np.ones_like(wq, dtype=np.int8))
        for j in np.flatnonzero(wmask):
            out_week = int(o['wip.out_week'][j])
            if out_week <= t + 8:
                key = (int(o['wip.node'][j]), int(o['wip.k'][j]))
                discount = 1.0 if out_week <= t + 4 else 0.6
                committed[key] = committed.get(key, 0.0) + max(0.0, float(wq[j])) * discount

        u = np.maximum(0.0, np.asarray(o['graph_now.u'], dtype=float))
        c = np.asarray(o['graph_now.c'], dtype=float)
        tariff = np.asarray(o['graph_now.tariff'], dtype=float)
        mask = np.asarray(o['action_mask'], dtype=bool)
        edge_left = u.copy()
        cp_left = {}
        for node, row in self.cp_rows.items():
            cp_left[(node, 'tb')] = max(0.0, float(o['graph_now.kappa.tb'][row]))
            cp_left[(node, 'ct')] = max(0.0, float(o['graph_now.kappa.ct'][row]))

        candidates = {}
        for i, (src, dst, k, es, nominal, cps) in enumerate(self.routes):
            if not mask[i] or nominal <= 0:
                continue
            lead = max(1.0, float(sum(max(0.0, tau[e]) for e in es)))
            if t + lead > self.T:
                continue
            passage = t
            unsafe = False
            for e in es:
                if pending.get((e, k), self.T + 100) <= passage:
                    unsafe = True
                    break
                passage += max(0.0, float(tau[e]))
            if unsafe:
                continue
            op = min([max(0.0, float(opened[self.cp_rows[node]])) for node in cps] or [1.0])
            cap = min([nominal] + [float(u[e]) for e in es]) * op
            if cap <= 1e-9:
                continue
            freight = sum(float(c[e]) + float(tariff[e, k]) * self.v[k] for e in es)
            # Include lead time and disruption risk in the generalized route cost.
            risk = sum(1.0 - max(0.0, float(opened[self.cp_rows[node]])) for node in cps)
            score = freight + 0.022 * self.v[k] * lead + 0.10 * self.v[k] * risk
            candidates.setdefault((dst, k), []).append((score, i, lead, cap))

        # Forecast-derived target stock at each destination, incorporating arrival-time demand.
        needs = []
        left = max(0, self.T - t + 1)
        for key, options in candidates.items():
            options.sort()
            lead = options[0][2]
            horizon = min(float(left), max(1.0, lead + 1.35))
            if key in self.dem_rows:
                j = self.dem_rows[key]
                forecast = np.asarray(o['demand_forecast.qty'][j], dtype=float)
                fm = o.get('demand_forecast.qty.observed')
                valid = np.ones(forecast.shape, dtype=bool) if fm is None else np.asarray(fm[j], dtype=bool)
                vals = forecast[valid]
                rate = max(0.0, float(np.mean(vals))) if len(vals) else self.last_rate.get(key, self.rate.get(key, 0.0))
                self.last_rate[key] = rate
                target = 0.0
                for h in range(min(len(forecast), int(np.ceil(horizon)))):
                    if h < horizon:
                        amount = max(0.0, float(forecast[h])) if valid[h] else rate
                        target += amount * min(1.0, horizon - h)
                if horizon > len(forecast):
                    target += rate * (horizon - len(forecast))
                target += max(0.0, float(o['backlog.qty'][j]))
                priority = 2.0
            else:
                rate = max(0.0, self.rate.get(key, 0.0))
                target = rate * horizon
                priority = 0.75
            have = stock.get(key, 0.0) + committed.get(key, 0.0)
            need = max(0.0, target - have)
            if need > 1e-9:
                needs.append((-priority * need / max(rate, 1e-8), key, need))

        for _, key, need in sorted(needs):
            for score, i, lead, cap in candidates[key]:
                if need <= 1e-9:
                    break
                src, dst, k, es, nominal, cps = self.routes[i]
                pool = self.pool[k] if k < len(self.pool) else 'ct'
                qty = min(need, cap, available.get((src, k), 0.0))
                qty = min([qty] + [edge_left[e] for e in es])
                for node in cps:
                    qty = min(qty, cp_left.get((node, pool), 0.0))
                if qty <= 1e-9:
                    continue
                flows[i] += qty
                available[(src, k)] = max(0.0, available.get((src, k), 0.0) - qty)
                need -= qty
                for e in es:
                    edge_left[e] = max(0.0, edge_left[e] - qty)
                for node in cps:
                    pkey = (node, pool)
                    cp_left[pkey] = max(0.0, cp_left.get(pkey, 0.0) - qty)
        return {'flows': flows}