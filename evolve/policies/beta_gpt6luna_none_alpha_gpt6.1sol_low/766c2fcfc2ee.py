# -2.108107930446612
import numpy as np

class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.T = int(config['T'])
        self.slots = self.s['action_slots']
        self.edges = self.s['edges']
        self.tail = np.asarray(self.edges['tail'], dtype=int)
        self.head = np.asarray(self.edges['head'], dtype=int)
        self.v = np.asarray(self.s['commodities']['v'], dtype=float)
        self.pool = self.s['commodities'].get('pool', [])
        self.cp_row = {int(n): i for i, n in enumerate(self.l.get('chokepoints', []))}
        self.stock_row = {tuple(map(int, x)): i for i, x in enumerate(self.l.get('stock_slots', []))}
        self.supply_row = {tuple(map(int, x)): i for i, x in enumerate(self.l.get('supply_slots', []))}
        self.demand_row = {tuple(map(int, x)): i for i, x in enumerate(self.l.get('demands', []))}
        sinks = self.s.get('sinks', {})
        self.penalty = np.asarray(sinks.get('pi', np.ones(len(self.l.get('demands', [])))), dtype=float)
        self.routes = []
        self.out_rate = {}
        for i, (edge, k, lane) in enumerate(zip(self.slots['edge'], self.slots['k'], self.slots['lane'])):
            edge, k = int(edge), int(k)
            es = [edge] if lane is None or int(lane) < 0 else list(map(int, self.s['lanes']['edges'][int(lane)]))
            if not es:
                continue
            src, dst = int(self.tail[es[0]]), int(self.head[es[-1]])
            caps = [float(self.edges['u0'][e]) for e in es if self.edges['u0'][e] is not None]
            nominal = min(caps) if caps else 0.0
            self.routes.append((src, dst, k, es, nominal, i))
            self.out_rate[(src, k)] = self.out_rate.get((src, k), 0.0) + nominal
        # Slot indices may be sparse only if malformed input; normal tables are aligned.
        self.nslots = len(self.slots['edge'])
        self.sink_keys = set(self.demand_row)

    def act(self, observation):
        o = observation
        week = int(o['week'][0])
        flows = np.zeros(self.nslots, dtype=float)
        stock = {key: max(0.0, float(o['stock.qty'][row])) for key, row in self.stock_row.items()}
        avail = dict(stock)
        for key, row in self.supply_row.items():
            avail[key] = avail.get(key, 0.0) + max(0.0, float(o['graph_now.supply.avail'][row]))

        # Count arrivals only if they can contribute to the near-term sink need.
        incoming = {}
        pq = o.get('pipeline.qty', np.zeros(0))
        pm = o.get('pipeline.qty.observed', np.ones_like(pq, dtype=np.int8))
        lane_mask = o.get('pipeline.lane.observed', np.ones_like(pm, dtype=np.int8))
        lane_dest = {}
        for lane, es0 in enumerate(self.s['lanes']['edges']):
            es = list(map(int, es0))
            if es:
                for e in es:
                    lane_dest[(e, lane)] = int(self.head[es[-1]])
        for j in np.flatnonzero(pm):
            q = max(0.0, float(pq[j]))
            if q <= 0:
                continue
            arr = int(o['pipeline.arrival_week'][j])
            if arr > week + 6:
                continue
            e = int(o['pipeline.edge'][j])
            k = int(o['pipeline.k'][j])
            lane = int(o['pipeline.lane'][j]) if lane_mask[j] else -1
            dst = lane_dest.get((e, lane), int(self.head[e]))
            key = (dst, k)
            incoming[key] = incoming.get(key, 0.0) + q
        # WIP counts only at a matching stocked node or sink; avoid counting unrelated late output.
        wq = o.get('wip.qty', np.zeros(0))
        wm = o.get('wip.qty.observed', np.ones_like(wq, dtype=np.int8))
        for j in np.flatnonzero(wm):
            if int(o['wip.out_week'][j]) <= week + 6:
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

        candidates = {}
        for src, dst, k, es, nominal, i in self.routes:
            if not mask[i] or nominal <= 0 or (dst, k) not in self.sink_keys:
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
            if op <= 1e-6:
                continue
            cap = min([nominal] + [max(0.0, float(u[e])) for e in es]) * op
            if cap <= 1e-9:
                continue
            freight = sum(float(c[e]) + float(tariff[e, k]) * self.v[k] for e in es)
            # Costs include an explicit time penalty, but shortage value dominates for urgent sinks.
            route_cost = freight + 0.02 * self.v[k] * max(0.0, lead) + (1.0 - op) * 0.1 * self.v[k]
            candidates.setdefault((dst, k), []).append((route_cost, i, lead, cap, cps))

        needs = []
        left_horizon = max(0, self.T - week + 1)
        forecast = np.asarray(o['demand_forecast.qty'], dtype=float)
        fmask = o.get('demand_forecast.qty.observed')
        for key, row in self.demand_row.items():
            routes = candidates.get(key, [])
            if not routes:
                continue
            vals = forecast[row]
            if fmask is not None:
                vals = vals[np.asarray(fmask[row], dtype=bool)]
            rate = max(0.0, float(np.mean(vals))) if len(vals) else 0.0
            lead = min(r[2] for r in routes)
            horizon = min(left_horizon, max(1.0, lead + 2.0))
            demand = rate * horizon
            backlog = max(0.0, float(o['backlog.qty'][row]))
            target = demand + backlog
            have = stock.get(key, 0.0) + incoming.get(key, 0.0)
            need = max(0.0, target - have)
            if need <= 1e-9:
                continue
            pi = float(self.penalty[row]) if row < len(self.penalty) else 1.0
            # Larger penalty, larger backlog, and shorter delivery lead raise priority.
            urgency = pi * (need + backlog) / max(1.0, rate * horizon)
            needs.append((-urgency, -pi, key, need))

        for _, _, key, need in sorted(needs):
            for cost, i, lead, cap, cps in sorted(candidates[key]):
                if need <= 1e-9:
                    break
                src, dst, k, es, nominal, _ = self.routes[i]
                qty = min(need, cap, avail.get((src, k), 0.0))
                if qty <= 0:
                    continue
                qty = min([qty] + [edge_left[e] for e in es])
                pool = self.pool[k] if k < len(self.pool) else 'ct'
                for node in cps:
                    qty = min(qty, cp_left.get((node, pool), 0.0))
                if qty <= 1e-9:
                    continue
                flows[i] = qty
                avail[(src, k)] = max(0.0, avail.get((src, k), 0.0) - qty)
                need -= qty
                for e in es:
                    edge_left[e] = max(0.0, edge_left[e] - qty)
                for node in cps:
                    cp_left[(node, pool)] = max(0.0, cp_left.get((node, pool), 0.0) - qty)
        return {'flows': flows}
