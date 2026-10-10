# 0.5674080508692851
import numpy as np

class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.T = config['T']
        s = self.s
        self.edges = s['edges']
        self.routes = []
        self.cp = {n: i for i, n in enumerate(self.l['chokepoints'])}
        a = s['action_slots']
        for i, (e, k, lane) in enumerate(zip(a['edge'], a['k'], a['lane'])):
            path = list(s['lanes']['edges'][lane]) if lane is not None and lane >= 0 else [e]
            cps = list(s['lanes']['chokepoints'][lane]) if lane is not None and lane >= 0 else []
            self.routes.append((i, e, k, lane, path, cps, self.edges['tail'][path[0]], self.edges['head'][path[-1]]))
        self.dest = {(r[7], r[2]) for r in self.routes}
        self.stockslots = self.l['stock_slots']
        self.demands = {tuple(x): i for i, x in enumerate(self.l['demands'])}
        self.sinks = {(s['sinks']['node'][i], s['sinks']['k'][i]): i for i in range(len(s['sinks']['node']))}
        self.backlog_sinks = set(i for i in range(len(s['sinks']['backlog'])) if s['sinks']['backlog'][i])
        self.sinks_pi = np.array([s['sinks']['pi'][i] for i in range(len(s['sinks']['pi']))])
        
        groups = {}
        incoming = {}
        for r in self.routes:
            i, e, k, lane, path, cps, tail, head = r
            cap = min(float(self.edges['u0'][p] or 0) for p in path)
            groups[(tail, head, k)] = max(groups.get((tail, head, k), 0), cap)
            incoming[(head, k)] = max(incoming.get((head, k), 0), cap)
        self.rate = {}
        for (tail, head, k), cap in groups.items():
            self.rate[(tail, k)] = self.rate.get((tail, k), 0) + cap
        for key in self.dest:
            if self.rate.get(key, 0) <= 0:
                self.rate[key] = incoming.get(key, 0)
        
        self.override = np.zeros(config['spaces']['action']['override_qty']['shape'])
        self.release = np.zeros(config['spaces']['action']['release_mode']['shape'], dtype=np.int64)

    def act(self, o):
        week = int(o['week'][0])
        remaining = max(0, self.T - week + 1)
        stock = {tuple(key): float(q) for key, q in zip(self.stockslots, o['stock.qty'])}
        available = dict(stock)
        for key, q in zip(self.l['supply_slots'], o['graph_now.supply.avail']):
            key = tuple(key)
            available[key] = available.get(key, 0) + float(q)
        
        position = dict(stock)
        live = o.get('pipeline.qty.observed', np.ones_like(o['pipeline.qty']))
        for j in np.flatnonzero(live):
            q = float(o['pipeline.qty'][j])
            if q <= 0:
                continue
            e = int(o['pipeline.edge'][j])
            k = int(o['pipeline.k'][j])
            head = self.edges['head'][e]
            lane = int(o['pipeline.lane'][j])
            arrival = int(o['pipeline.arrival_week'][j])
            lm = o.get('pipeline.lane.observed')
            if (lm is None or lm[j]) and 0 <= lane < len(self.s['lanes']['edges']):
                head = self.edges['head'][self.s['lanes']['edges'][lane][-1]]
            key = (head, k)
            delay_to_arrival = max(0, arrival - week)
            if delay_to_arrival >= remaining:
                weight = 0.0
            elif delay_to_arrival == 0:
                weight = 1.0
            else:
                weight = min(1.0, max(0.2, (remaining - delay_to_arrival) / max(remaining, 1.0)))
            position[key] = position.get(key, 0) + q * weight
        
        if 'queue_lots.qty' in o:
            for key, row in zip(self.l.get('lot_keys', []), o['queue_lots.qty']):
                cp, k, lane, e = key
                head = self.edges['head'][e]
                if lane is not None and lane >= 0:
                    head = self.edges['head'][self.s['lanes']['edges'][lane][-1]]
                ix = self.cp.get(cp)
                if ix is not None:
                    opened = float(o['graph_now.open'][ix])
                    weight = 0.15 + 0.85 * opened
                else:
                    weight = 1.0
                pair = (head, k)
                position[pair] = position.get(pair, 0) + float(np.sum(row)) * weight
        
        if 'wip.qty' in o:
            mask = o.get('wip.qty.observed', np.ones_like(o['wip.qty']))
            for j in np.flatnonzero(mask):
                out_week = int(o['wip.out_week'][j])
                if out_week <= self.T:
                    pair = (int(o['wip.node'][j]), int(o['wip.k'][j]))
                    position[pair] = position.get(pair, 0) + float(o['wip.qty'][j])
        
        forecast = o['demand_forecast.qty']
        rates = dict(self.rate)
        for pair, j in self.demands.items():
            rates[pair] = float(np.mean(forecast[j, :min(4, forecast.shape[1])]))
        
        u = o['graph_now.u']
        costs = o['graph_now.c']
        tau = o['graph_now.tau']
        tariffs = o['graph_now.tariff']
        values = self.s['commodities']['v']
        open_fractions = o['graph_now.open']
        war_risk = o['graph_now.war_risk']
        warning_scores = o.get('warning.score', np.zeros(1))
        max_warning = float(np.max(warning_scores)) if len(warning_scores) > 0 else 0.0
        
        pending = {}
        if 'pending_prohibitions.edge' in o:
            mask = o.get('pending_prohibitions.edge.observed', np.ones_like(o['pending_prohibitions.edge']))
            for j in np.flatnonzero(mask):
                edge_idx = int(o['pending_prohibitions.edge'][j])
                k_idx = int(o['pending_prohibitions.k'][j])
                eff_week = int(o['pending_prohibitions.effective_week'][j])
                if eff_week >= week:
                    pending[(edge_idx, k_idx)] = min(pending.get((edge_idx, k_idx), eff_week), eff_week)
        
        options = []
        minlead = {}
        for r in self.routes:
            i, e, k, lane, path, cps, tail, head = r
            if not o['action_mask'][i]:
                continue
            cap = min(float(u[p]) for p in path)
            lead = sum(max(0, int(tau[p])) for p in path)
            closure = 0.0
            closure_multiplier = 1.0
            war_penalty = 0.0
            
            for cp in cps:
                ix = self.cp.get(cp)
                if ix is not None:
                    opened = float(open_fractions[ix])
                    closure += 7.5 * (1 - opened) ** 1.2
                    closure_multiplier *= (0.3 + 0.7 * opened)
                    pool = self.s['commodities']['pool'][k]
                    field = 'graph_now.kappa.' + str(pool)
                    if field in o:
                        cap = min(cap, float(o[field][ix]) * opened)
                    risk = int(war_risk[ix])
                    if risk > 0:
                        war_penalty += 0.25 * risk * (1 - opened)
            
            prohibition_risk = 0.0
            elapsed = 0
            for p in path:
                if pending.get((p, k), self.T + 100) <= week + elapsed:
                    prohibition_risk += 2.2
                elapsed += max(0, int(tau[p]))
            
            if cap <= 0 or lead >= remaining:
                continue
            pair = (head, k)
            minlead[pair] = min(minlead.get(pair, 1e9), lead + closure)
            v = float(values[k])
            expense = sum(float(costs[p]) + float(tariffs[p, k]) * v for p in path)
            base_score = (expense / max(v, 1.0)) * closure_multiplier + 0.036 * lead + 0.175 * closure + war_penalty + prohibition_risk
            warning_adjustment = 1.0 + 0.20 * max_warning
            score = base_score * warning_adjustment
            options.append((score, r, cap, lead))
        
        flows = np.zeros(len(self.routes), dtype=float)
        edge_used = np.zeros(len(u))
        targets = {}
        
        for pair in self.dest:
            lead = minlead.get(pair, 0)
            horizon = min(remaining, max(1.5, lead + 2.0))
            target = rates.get(pair, 0) * horizon
            
            if pair in self.demands:
                j = self.demands[pair]
                sink_idx = self.sinks.get(pair)
                n = min(forecast.shape[1], max(1, int(np.ceil(horizon))))
                target = float(np.sum(forecast[j, :n])) + max(0, horizon - n) * rates[pair]
                backlog = float(o['backlog.qty'][j])
                target += backlog
                if sink_idx in self.backlog_sinks:
                    pi = float(self.sinks_pi[j])
                    target *= (1.2 + 0.08 * min(backlog / max(rates[pair], 1.0), 5.0))
            
            safety_mult = 1.0 + 0.15 * max_warning
            targets[pair] = target * safety_mult
        
        for score, r, cap, lead in sorted(options, key=lambda x: x[0]):
            i, e, k, lane, path, cps, tail, head = r
            pair = (head, k)
            origin = (tail, k)
            deficit = max(0, targets.get(pair, 0) - position.get(pair, 0))
            room = min(max(0, float(u[p]) - edge_used[p]) for p in path)
            qty = min(cap, room, deficit, max(0, available.get(origin, 0)))
            if qty <= 0:
                continue
            flows[i] = qty
            available[origin] = available.get(origin, 0) - qty
            position[pair] = position.get(pair, 0) + qty
            for p in path:
                edge_used[p] += qty
        
        return {'flows': flows, 'override_qty': self.override, 'release_mode': self.release}
