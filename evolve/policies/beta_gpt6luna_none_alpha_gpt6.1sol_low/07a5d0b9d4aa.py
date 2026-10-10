# 0.5411596850310655
import numpy as np

class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.T = int(config['T'])
        s = self.s
        a = s['action_slots']
        self.n = len(a['edge'])
        self.edges = s['edges']
        self.tail = self.edges['tail']
        self.head = self.edges['head']
        self.routes = []
        self.groups = {}
        self.cp = {int(n): i for i, n in enumerate(self.l['chokepoints'])}
        self.supply = {tuple(map(int, x)): i for i, x in enumerate(self.l['supply_slots'])}
        self.dem = {tuple(map(int, x)): i for i, x in enumerate(self.l['demands'])}
        self.rate = {}
        outgoing = {}
        for i, (e, k, lane) in enumerate(zip(a['edge'], a['k'], a['lane'])):
            e, k = int(e), int(k)
            es = [e] if lane is None or int(lane) < 0 else list(s['lanes']['edges'][int(lane)])
            src, dst = int(self.tail[es[0]]), int(self.head[es[-1]])
            caps = [float(self.edges['u0'][j]) for j in es if self.edges['u0'][j] is not None]
            cap = min(caps) if caps else 0.0
            self.routes.append((src, dst, k, es, cap))
            self.groups.setdefault((dst, k), []).append(i)
            key = (src, dst, k)
            outgoing[key] = max(outgoing.get(key, 0.0), cap)
        for (src, dst, k), cap in outgoing.items():
            self.rate[(src, k)] = self.rate.get((src, k), 0.0) + cap
        for key, ids in self.groups.items():
            if key not in self.rate:
                self.rate[key] = max(self.routes[i][4] for i in ids)
        self.edge_dest = {}
        for lane, es in enumerate(s['lanes']['edges']):
            for e in es:
                self.edge_dest[(int(e), lane)] = int(self.head[es[-1]])
        self.v = np.asarray(s['commodities']['v'], dtype=float)

    def act(self, observation):
        o = observation
        t = int(o['week'][0])
        stock = {tuple(map(int, key)): float(q) for key, q in zip(self.l['stock_slots'], o['stock.qty'])}
        committed = {}
        qmask = o.get('pipeline.qty.observed', np.ones_like(o['pipeline.qty']))
        for j in np.flatnonzero(qmask):
            q = float(o['pipeline.qty'][j])
            if q <= 0:
                continue
            e, k = int(o['pipeline.edge'][j]), int(o['pipeline.k'][j])
            lane = int(o['pipeline.lane'][j]) if o.get('pipeline.lane.observed', np.ones_like(qmask))[j] else -1
            dst = self.edge_dest.get((e, lane), int(self.head[e]))
            key = (dst, k)
            committed[key] = committed.get(key, 0.0) + q
        lots = o.get('queue_lots.qty')
        if lots is not None:
            for row, key in enumerate(self.l.get('lot_keys', [])):
                cp, k, lane, edge = map(int, key)
                dst = self.edge_dest.get((edge, lane), int(self.head[edge]))
                pair = (dst, k)
                committed[pair] = committed.get(pair, 0.0) + float(np.sum(lots[row]))
        wm = o.get('wip.qty.observed', np.zeros_like(o.get('wip.qty', [])))
        for j in np.flatnonzero(wm):
            key = (int(o['wip.node'][j]), int(o['wip.k'][j]))
            committed[key] = committed.get(key, 0.0) + float(o['wip.qty'][j])
        u = np.asarray(o['graph_now.u'], dtype=float)
        c = np.asarray(o['graph_now.c'], dtype=float)
        tau = np.asarray(o['graph_now.tau'], dtype=float)
        tariff = o['graph_now.tariff']
        mask = o['action_mask']
        candidates = {}
        edge_left = np.maximum(0.0, u.copy())
        for i, (src, dst, k, es, nominal) in enumerate(self.routes):
            if not mask[i]:
                continue
            lead = max(1.0, float(sum(max(0.0, tau[e]) for e in es)))
            if t + lead > self.T:
                continue
            cap = min([max(0.0, float(u[e])) for e in es] + [nominal])
            openness = 1.0
            for e in es:
                node = int(self.head[e])
                if node in self.cp:
                    openness = min(openness, float(o['graph_now.open'][self.cp[node]]))
            if openness <= 0.01 or cap <= 0:
                continue
            cap *= openness
            cost = sum(float(c[e]) + float(tariff[e, k]) * self.v[k] for e in es)
            score = cost + 0.025 * self.v[k] * lead + (1.0 - openness) * 0.15 * self.v[k]
            candidates.setdefault((dst, k), []).append((score, i, lead, cap))
        flows = np.zeros(self.n, dtype=float)
        source_left = {}
        for key, q in stock.items():
            source_left[key] = max(0.0, q)
        for key, j in self.supply.items():
            source_left[key] = source_left.get(key, 0.0) + max(0.0, float(o['graph_now.supply.avail'][j]))
        priorities = []
        for key, routes in candidates.items():
            lead = min(x[2] for x in routes)
            remaining = max(0, self.T - t + 1)
            if key in self.dem:
                j = self.dem[key]
                forecast = np.asarray(o['demand_forecast.qty'][j], dtype=float)
                valid = o.get('demand_forecast.qty.observed')
                if valid is not None:
                    vals = forecast[np.asarray(valid[j], dtype=bool)]
                else:
                    vals = forecast
                rate = max(0.0, float(np.mean(vals))) if len(vals) else self.rate.get(key, 0.0)
                horizon = min(remaining, lead + 1.5)
                target = rate * horizon + float(o['backlog.qty'][j])
                priority = 2.0
            else:
                rate = self.rate.get(key, 0.0)
                target = rate * min(remaining, lead + 1.5)
                priority = 1.0
            have = stock.get(key, 0.0) + committed.get(key, 0.0)
            need = max(0.0, target - have)
            urgency = priority * need / max(rate, 1e-9)
            priorities.append((-urgency, key, need))
        for _, key, need in sorted(priorities):
            for score, i, lead, cap in sorted(candidates[key]):
                if need <= 0:
                    break
                src, dst, k, es, nominal = self.routes[i]
                available = source_left.get((src, k), 0.0)
                qty = min(need, cap, available, min(edge_left[e] for e in es))
                if qty <= 0:
                    continue
                flows[i] = qty
                source_left[(src, k)] = available - qty
                need -= qty
                for e in es:
                    edge_left[e] = max(0.0, edge_left[e] - qty)
        return {'flows': flows}
