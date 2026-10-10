# 0.5576484044616816
import numpy as np

class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.T = config['T']
        self.e = self.s['edges']
        self.cp = {n: j for j, n in enumerate(self.l['chokepoints'])}
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
        self.cache = {}

    def graph(self, o, key, nominal):
        x = np.asarray(o[key]).copy()
        mask = o.get(key + '.observed', np.ones_like(x))
        base = self.cache.get(key, np.asarray(nominal))
        x = np.where(mask, x, base)
        self.cache[key] = x.copy()
        return x

    def act(self, o):
        week = int(o['week'][0])
        remaining = max(0, self.T - week + 1)
        stock = {tuple(p): max(0., float(q)) for p, q in zip(self.l['stock_slots'], o['stock.qty'])}
        available = dict(stock)
        for p, q in zip(self.l['supply_slots'], o['graph_now.supply.avail']):
            p = tuple(p)
            available[p] = available.get(p, 0) + max(0., float(q))
        position = dict(stock)
        u = self.graph(o, 'graph_now.u', [float(x or 0) for x in self.e['u0']])
        c = self.graph(o, 'graph_now.c', self.e['c0'])
        tau = self.graph(o, 'graph_now.tau', self.e['tau0'])
        opened = self.graph(o, 'graph_now.open', np.ones(len(self.cp)))
        warnings = {tuple(p): max(0., float(q)) for p, q in zip(self.l['warning_units'], o.get('warning.score', []))}
        max_warn = max(warnings.values(), default=0.)
        pending = {}
        for j in np.flatnonzero(o.get('pending_prohibitions.edge.observed', np.zeros_like(o.get('pending_prohibitions.edge', [])))):
            pair = (int(o['pending_prohibitions.edge'][j]), int(o['pending_prohibitions.k'][j]))
            w = int(o['pending_prohibitions.effective_week'][j])
            if w >= week:
                pending[pair] = min(pending.get(pair, w), w)
        ends = {}
        for j in np.flatnonzero(o.get('closure_end.end_week.observed', np.zeros_like(o.get('closure_end.end_week', [])))):
            cp = int(o['closure_end.chokepoint'][j])
            ends[cp] = max(ends.get(cp, week), int(o['closure_end.end_week'][j]))
        options, minlead = [], {}
        for r in self.routes:
            i, k, path, cps, tail, head = r
            if not o['action_mask'][i]:
                continue
            cap = min(max(0., float(u[p])) for p in path)
            lead = sum(max(0, int(tau[p])) for p in path)
            closure, war, local = 0., 0., 0.
            for cp in cps:
                ix = self.cp.get(cp)
                if ix is not None:
                    op = float(opened[ix])
                    closure += 9. * (1 - op)
                    field = 'graph_now.kappa.' + str(self.s['commodities']['pool'][k])
                    if field in o:
                        cap = min(cap, max(0., float(o[field][ix]) * op))
                    war += .4 * int(o['graph_now.war_risk'][ix])
                    local = max(local, warnings.get(('chokepoint', cp), 0.))
            elapsed, blocked = 0, False
            for p in path:
                local = max(local, warnings.get(('region', self.s['nodes']['region'][self.e['tail'][p]]), 0.))
                if pending.get((p, k), self.T + 100) <= week + elapsed:
                    blocked = True
                elapsed += max(0, int(tau[p]))
            if blocked or cap <= 0 or lead >= remaining:
                continue
            pair = (head, k)
            minlead[pair] = min(minlead.get(pair, float('inf')), lead + closure)
            value = float(self.s['commodities']['v'][k])
            expense = sum(float(c[p]) + float(o['graph_now.tariff'][p, k]) * value for p in path)
            score = expense / max(value, 1.) + .034 * lead + .19 * closure + war + .12 * local
            options.append((score, r, cap, lead))
        arrivals = {}
        for j in np.flatnonzero(o.get('pipeline.qty.observed', np.ones_like(o['pipeline.qty']))):
            q = float(o['pipeline.qty'][j])
            if q <= 0:
                continue
            e, k = int(o['pipeline.edge'][j]), int(o['pipeline.k'][j])
            head = self.e['head'][e]
            lane = int(o['pipeline.lane'][j])
            lm = o.get('pipeline.lane.observed')
            delay = max(0, int(o['pipeline.arrival_week'][j]) - week)
            if (lm is None or lm[j]) and 0 <= lane < len(self.s['lanes']['edges']):
                path = self.s['lanes']['edges'][lane]
                head = self.e['head'][path[-1]]
                if e in path:
                    rest = path[path.index(e) + 1:]
                    delay += sum(max(0, int(tau[p])) for p in rest)
                    for cp in self.s['lanes']['chokepoints'][lane]:
                        if any(self.e['tail'][p] == cp for p in rest):
                            ix = self.cp.get(cp)
                            if ix is not None:
                                delay += max(0, ends.get(cp, week + 6) - week) * (1 - float(opened[ix]))
            pair = (head, k)
            weight = max(0., (remaining - delay) / max(1, remaining))
            position[pair] = position.get(pair, 0) + q * weight
            arrivals.setdefault(pair, []).append((delay, q))
        for key, row in zip(self.l.get('lot_keys', []), o.get('queue_lots.qty', [])):
            cp, k, lane, e = key
            head = self.e['head'][e]
            if lane is not None and lane >= 0:
                head = self.e['head'][self.s['lanes']['edges'][lane][-1]]
            ix = self.cp.get(cp)
            weight = .15 + .85 * float(opened[ix]) if ix is not None else 1.
            pair = (head, k)
            position[pair] = position.get(pair, 0) + float(np.sum(row)) * weight
        for j in np.flatnonzero(o.get('wip.qty.observed', np.zeros_like(o.get('wip.qty', [])))):
            if int(o['wip.out_week'][j]) <= self.T:
                pair = (int(o['wip.node'][j]), int(o['wip.k'][j]))
                position[pair] = position.get(pair, 0) + float(o['wip.qty'][j])
        forecast = o['demand_forecast.qty']
        rates, targets = dict(self.rate), {}
        for pair in self.dest:
            horizon = min(remaining, max(1., minlead.get(pair, 0) + 3.2))
            rate = rates.get(pair, 0)
            target = rate * horizon
            if pair in self.demands:
                j = self.demands[pair]
                rate = float(np.mean(forecast[j, :min(4, forecast.shape[1])]))
                rates[pair] = rate
                n = min(forecast.shape[1], int(np.ceil(horizon)))
                target = float(np.sum(forecast[j, :n])) + max(0, horizon - n) * rate
                target += float(o['backlog.qty'][j])
                if self.backlog.get(pair, False):
                    target *= 1.2
            targets[pair] = target * (1 + .15 * max_warn)
        flows = np.zeros(len(self.routes))
        used = np.zeros(len(u))
        def priority(option):
            score, r, cap, lead = option
            pair = (r[5], r[1])
            rate = rates.get(pair, 0)
            coverage = stock.get(pair, 0) / max(rate, 1e-9)
            urgency = .10 * max(0., 2. - coverage)
            if pair in self.demands and rate > 0:
                incoming = sum(q for delay, q in arrivals.get(pair, []) if delay <= lead)
                gap = max(0., rate * lead - stock.get(pair, 0) - incoming)
                score += .065 * min(lead, gap / rate)
            return score - urgency
        for score, r, cap, lead in sorted(options, key=priority):
            i, k, path, cps, tail, head = r
            pair, origin = (head, k), (tail, k)
            deficit = max(0., targets.get(pair, 0) - position.get(pair, 0))
            room = min(max(0., float(u[p]) - used[p]) for p in path)
            qty = min(cap, room, deficit, max(0., available.get(origin, 0)))
            if qty > 0 and np.isfinite(qty):
                flows[i] = qty
                available[origin] = available.get(origin, 0) - qty
                position[pair] = position.get(pair, 0) + qty
                for p in path:
                    used[p] += qty
        return {'flows': flows}
