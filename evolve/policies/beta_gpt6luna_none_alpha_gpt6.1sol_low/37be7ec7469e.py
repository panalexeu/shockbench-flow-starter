# 0.5422785429790424
import numpy as np

class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.T = int(config['T'])
        s = self.s
        self.e = s['edges']
        self.head = self.e['head']
        self.cp = {int(n): i for i, n in enumerate(self.l['chokepoints'])}
        self.dem = {tuple(map(int, x)): i for i, x in enumerate(self.l['demands'])}
        self.sup = {tuple(map(int, x)): i for i, x in enumerate(self.l['supply_slots'])}
        self.v = np.asarray(s['commodities']['v'], float)
        self.routes = []
        outgoing = {}
        incoming = {}
        a = s['action_slots']
        for e, k, lane in zip(a['edge'], a['k'], a['lane']):
            e, k = int(e), int(k)
            es = [e] if lane is None or int(lane) < 0 else list(map(int, s['lanes']['edges'][int(lane)]))
            src, dst = int(self.e['tail'][es[0]]), int(self.head[es[-1]])
            caps = [float(self.e['u0'][j]) for j in es if self.e['u0'][j] is not None]
            cap = min(caps) if caps else 0.0
            self.routes.append((src, dst, k, es, cap))
            outgoing[(src, dst, k)] = max(outgoing.get((src, dst, k), 0.0), cap)
            incoming[(dst, k)] = max(incoming.get((dst, k), 0.0), cap)
        self.rate = {}
        for (src, dst, k), cap in outgoing.items():
            self.rate[(src, k)] = self.rate.get((src, k), 0.0) + cap
        for key, cap in incoming.items():
            self.rate.setdefault(key, cap)
        self.dest = {}
        for lane, es in enumerate(s['lanes']['edges']):
            for e in es:
                self.dest[(int(e), lane)] = int(self.head[es[-1]])
        self.last_rate = {}

    def act(self, o):
        t = int(o['week'][0])
        stock = {tuple(map(int, key)): max(0.0, float(q)) for key, q in zip(self.l['stock_slots'], o['stock.qty'])}
        committed = {}
        opened = o['graph_now.open']
        tau = np.asarray(o['graph_now.tau'], float)
        def add(key, q):
            committed[key] = committed.get(key, 0.0) + max(0.0, q)
        pq = o['pipeline.qty']
        pm = o.get('pipeline.qty.observed', np.ones_like(pq))
        lm = o.get('pipeline.lane.observed', np.ones_like(pm))
        for j in np.flatnonzero(pm):
            e, k = int(o['pipeline.edge'][j]), int(o['pipeline.k'][j])
            lane = int(o['pipeline.lane'][j]) if lm[j] else -1
            dst = self.dest.get((e, lane), int(self.head[e]))
            discount = 1.0
            if lane >= 0:
                for node in self.s['lanes']['chokepoints'][lane]:
                    if float(opened[self.cp[int(node)]]) <= 0.01:
                        discount = 0.65
            add((dst, k), float(pq[j]) * discount)
        lots = o.get('queue_lots.qty')
        if lots is not None:
            for row, key in enumerate(self.l.get('lot_keys', [])):
                cp, k, lane, e = map(int, key)
                discount = 0.65 if float(opened[self.cp[cp]]) <= 0.01 else 1.0
                add((self.dest.get((e, lane), int(self.head[e])), k), float(np.sum(lots[row])) * discount)
        wq = o.get('wip.qty', np.zeros(0))
        for j in np.flatnonzero(o.get('wip.qty.observed', np.zeros_like(wq))):
            add((int(o['wip.node'][j]), int(o['wip.k'][j])), float(wq[j]))
        pending = {}
        pm = o.get('pending_prohibitions.edge.observed', [])
        ew = o.get('pending_prohibitions.effective_week.observed', np.ones_like(pm))
        for j in np.flatnonzero(pm):
            if ew[j]:
                key = (int(o['pending_prohibitions.edge'][j]), int(o['pending_prohibitions.k'][j]))
                pending[key] = min(pending.get(key, self.T + 100), int(o['pending_prohibitions.effective_week'][j]))
        u = np.maximum(0.0, np.asarray(o['graph_now.u'], float))
        c = np.asarray(o['graph_now.c'], float)
        tariff = o['graph_now.tariff']
        choices = {}
        for i, (src, dst, k, es, nominal) in enumerate(self.routes):
            if not o['action_mask'][i]:
                continue
            lead = max(1.0, sum(max(0.0, tau[e]) for e in es))
            if t + lead > self.T:
                continue
            passage = t
            unsafe = False
            for e in es:
                if pending.get((e, k), self.T + 100) <= passage:
                    unsafe = True
                    break
                passage += max(0.0, tau[e])
            if unsafe:
                continue
            op = 1.0
            for e in es:
                node = int(self.head[e])
                if node in self.cp:
                    op = min(op, float(opened[self.cp[node]]))
            cap = min([nominal] + [u[e] for e in es]) * op
            if cap <= 0.0:
                continue
            cost = sum(float(c[e]) + float(tariff[e, k]) * self.v[k] for e in es)
            score = cost + 0.025 * self.v[k] * lead + (1.0 - op) * 0.15 * self.v[k]
            choices.setdefault((dst, k), []).append((score, i, lead, cap))
        available = dict(stock)
        for key, j in self.sup.items():
            available[key] = available.get(key, 0.0) + max(0.0, float(o['graph_now.supply.avail'][j]))
        needs = []
        for key, options in choices.items():
            lead = min(options)[2]
            remaining = max(0, self.T - t + 1)
            if key in self.dem:
                j = self.dem[key]
                forecast = np.asarray(o['demand_forecast.qty'][j], float)
                fm = o.get('demand_forecast.qty.observed')
                vals = forecast if fm is None else forecast[np.asarray(fm[j], bool)]
                rate = max(0.0, float(np.mean(vals))) if len(vals) else self.last_rate.get(key, self.rate.get(key, 0.0))
                self.last_rate[key] = rate
                target = rate * min(remaining, lead + 2.25) + max(0.0, float(o['backlog.qty'][j]))
                priority = 2.0
            else:
                rate = self.rate.get(key, 0.0)
                target = rate * min(remaining, lead + 1.5)
                priority = 1.0
            need = max(0.0, target - stock.get(key, 0.0) - committed.get(key, 0.0))
            needs.append((-priority * need / max(rate, 1e-9), key, need))
        flows = np.zeros(len(self.routes), float)
        edge_left = u.copy()
        for _, key, need in sorted(needs):
            for score, i, lead, cap in sorted(choices[key]):
                if need <= 0:
                    break
                src, dst, k, es, nominal = self.routes[i]
                q = min(need, cap, available.get((src, k), 0.0), min(edge_left[e] for e in es))
                if q <= 0:
                    continue
                flows[i] = q
                available[(src, k)] -= q
                need -= q
                for e in es:
                    edge_left[e] = max(0.0, edge_left[e] - q)
        return {'flows': flows}
