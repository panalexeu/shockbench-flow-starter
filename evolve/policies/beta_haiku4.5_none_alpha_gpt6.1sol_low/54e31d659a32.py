# 0.5064144995338259
import numpy as np

class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.T = config['T']
        self.e = self.s['edges']
        self.cp = {n: i for i, n in enumerate(self.l['chokepoints'])}
        self.routes = []
        groups, incoming = {}, {}
        a = self.s['action_slots']
        for i, (e, k, lane) in enumerate(zip(a['edge'], a['k'], a['lane'])):
            path = list(self.s['lanes']['edges'][lane]) if lane is not None and lane >= 0 else [e]
            cps = list(self.s['lanes']['chokepoints'][lane]) if lane is not None and lane >= 0 else []
            tail, head = self.e['tail'][path[0]], self.e['head'][path[-1]]
            self.routes.append((i, k, path, cps, tail, head))
            cap = min(float(self.e['u0'][p] or 0) for p in path)
            groups[(tail, head, k)] = max(groups.get((tail, head, k), 0), cap)
            incoming[(head, k)] = max(incoming.get((head, k), 0), cap)
        self.dest = {(r[5], r[1]) for r in self.routes}
        self.rate = {}
        for (tail, head, k), cap in groups.items():
            self.rate[(tail, k)] = self.rate.get((tail, k), 0) + cap
        for pair in self.dest:
            if self.rate.get(pair, 0) <= 0:
                self.rate[pair] = incoming.get(pair, 0)
        self.demands = {tuple(p): j for j, p in enumerate(self.l['demands'])}
        self.backlog = {(n, k): bool(b) for n, k, b in zip(self.s['sinks']['node'], self.s['sinks']['k'], self.s['sinks']['backlog'])}

    def act(self, o):
        week = int(o['week'][0])
        remaining = max(0, self.T - week + 1)
        stock = {tuple(p): float(q) for p, q in zip(self.l['stock_slots'], o['stock.qty'])}
        available = dict(stock)
        for p, q in zip(self.l['supply_slots'], o['graph_now.supply.avail']):
            p = tuple(p)
            available[p] = available.get(p, 0) + float(q)
        position = dict(stock)
        near = dict(stock)
        u, c, tau = o['graph_now.u'], o['graph_now.c'], o['graph_now.tau']
        opened = o['graph_now.open']
        live = o.get('pipeline.qty.observed', np.ones_like(o['pipeline.qty']))
        for j in np.flatnonzero(live):
            q = float(o['pipeline.qty'][j])
            if q <= 0:
                continue
            e, k = int(o['pipeline.edge'][j]), int(o['pipeline.k'][j])
            head = self.e['head'][e]
            lane = int(o['pipeline.lane'][j])
            lm = o.get('pipeline.lane.observed')
            delay = max(0, int(o['pipeline.arrival_week'][j]) - week)
            total_delay = delay
            if (lm is None or lm[j]) and 0 <= lane < len(self.s['lanes']['edges']):
                path = self.s['lanes']['edges'][lane]
                head = self.e['head'][path[-1]]
                if e in path:
                    total_delay += sum(max(0, int(tau[p])) for p in path[path.index(e) + 1:])
            pair = (head, k)
            weight = 0.0 if total_delay >= remaining else (0.95 if delay <= 1 else max(0.25, 0.95 - 0.15 * delay))
            position[pair] = position.get(pair, 0) + q * weight
            if total_delay <= 1:
                near[pair] = near.get(pair, 0) + q
        for key, row in zip(self.l.get('lot_keys', []), o.get('queue_lots.qty', [])):
            cp, k, lane, e = key
            head = self.e['head'][e]
            if lane is not None and lane >= 0:
                head = self.e['head'][self.s['lanes']['edges'][lane][-1]]
            ix = self.cp.get(cp)
            weight = 0.18 + 0.82 * float(opened[ix]) if ix is not None else 1.0
            pair = (head, k)
            position[pair] = position.get(pair, 0) + float(np.sum(row)) * weight
        if 'wip.qty' in o:
            mask = o.get('wip.qty.observed', np.ones_like(o['wip.qty']))
            for j in np.flatnonzero(mask):
                if int(o['wip.out_week'][j]) <= self.T:
                    pair = (int(o['wip.node'][j]), int(o['wip.k'][j]))
                    position[pair] = position.get(pair, 0) + float(o['wip.qty'][j])
        forecast = o['demand_forecast.qty']
        rates = dict(self.rate)
        for pair, j in self.demands.items():
            rates[pair] = float(np.mean(forecast[j, :min(4, forecast.shape[1])]))
        warning = o.get('warning.score', np.zeros(1))
        max_warning = max(0.0, float(np.max(warning))) if len(warning) else 0.0
        pending = {}
        if 'pending_prohibitions.edge' in o:
            mask = o.get('pending_prohibitions.edge.observed', np.ones_like(o['pending_prohibitions.edge']))
            for j in np.flatnonzero(mask):
                pair = (int(o['pending_prohibitions.edge'][j]), int(o['pending_prohibitions.k'][j]))
                w = int(o['pending_prohibitions.effective_week'][j])
                if w >= week:
                    pending[pair] = min(pending.get(pair, w), w)
        options, minlead = [], {}
        for r in self.routes:
            i, k, path, cps, tail, head = r
            if not o['action_mask'][i]:
                continue
            cap = min(float(u[p]) for p in path)
            lead = sum(max(0, int(tau[p])) for p in path)
            closure, war = 0.0, 0.0
            for cp in cps:
                ix = self.cp.get(cp)
                if ix is not None:
                    op = float(opened[ix])
                    closure += 8.2 * (1 - op)
                    field = 'graph_now.kappa.' + str(self.s['commodities']['pool'][k])
                    if field in o:
                        cap = min(cap, float(o[field][ix]) * op)
                    war += 0.32 * int(o['graph_now.war_risk'][ix])
            if cap <= 0 or lead >= remaining:
                continue
            pair = (head, k)
            minlead[pair] = min(minlead.get(pair, float('inf')), lead + closure)
            value = float(self.s['commodities']['v'][k])
            expense = sum(float(c[p]) + float(o['graph_now.tariff'][p, k]) * value for p in path)
            elapsed, risk = 0, 0.0
            for p in path:
                if pending.get((p, k), self.T + 100) <= week + elapsed:
                    risk += 2.4
                elapsed += max(0, int(tau[p]))
            rate = rates.get(pair, 0)
            coverage = near.get(pair, 0) / max(rate, 1e-9)
            urgency = max(0.0, 1.0 - coverage / max(lead + 1, 1))
            if pair in self.demands:
                urgency = min(1.0, urgency + float(o['backlog.qty'][self.demands[pair]]) / max(rate * 4, 1e-9))
            score = expense / max(value, 1.0) + (0.027 + 0.022 * urgency) * lead + 0.188 * closure + war + risk
            options.append((score, r, cap, lead))
        targets = {}
        for pair in self.dest:
            horizon = min(remaining, max(1.2, minlead.get(pair, 0) + 2.2))
            target = rates.get(pair, 0) * horizon
            if pair in self.demands:
                j = self.demands[pair]
                n = min(forecast.shape[1], max(1, int(np.ceil(horizon))))
                target = float(np.sum(forecast[j, :n])) + max(0, horizon - n) * rates[pair]
                target += float(o['backlog.qty'][j])
                if self.backlog.get(pair, False):
                    target *= 1.20
            targets[pair] = target * (1 + 0.17 * max_warning)
        flows = np.zeros(len(self.routes), dtype=float)
        used = np.zeros(len(u))
        for score, r, cap, lead in sorted(options, key=lambda x: x[0]):
            i, k, path, cps, tail, head = r
            pair, origin = (head, k), (tail, k)
            deficit = max(0, targets.get(pair, 0) - position.get(pair, 0))
            room = min(max(0, float(u[p]) - used[p]) for p in path)
            qty = min(cap, room, deficit, max(0, available.get(origin, 0)))
            if qty > 0 and np.isfinite(qty):
                flows[i] = qty
                available[origin] = available.get(origin, 0) - qty
                position[pair] = position.get(pair, 0) + qty
                for p in path:
                    used[p] += qty
        return {'flows': flows}
