# 0.51093939828247
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
        self.override_slots_list = config['static']['override_slots']
        self.tanker_commodities = set(i for i in range(len(s['commodities']['override'])) if s['commodities']['override'][i])
        
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
            lm = o.get('pipeline.lane.observed')
            if (lm is None or lm[j]) and 0 <= lane < len(self.s['lanes']['edges']):
                head = self.edges['head'][self.s['lanes']['edges'][lane][-1]]
            key = (head, k)
            position[key] = position.get(key, 0) + q
        
        if 'queue_lots.qty' in o:
            for key, row in zip(self.l.get('lot_keys', []), o['queue_lots.qty']):
                cp, k, lane, e = key
                head = self.edges['head'][e]
                if lane is not None and lane >= 0:
                    head = self.edges['head'][self.s['lanes']['edges'][lane][-1]]
                pair = (head, k)
                position[pair] = position.get(pair, 0) + float(np.sum(row))
        
        if 'wip.qty' in o:
            mask = o.get('wip.qty.observed', np.ones_like(o['wip.qty']))
            for j in np.flatnonzero(mask):
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
        warning_scores = o['warning.score']
        
        max_warning = float(np.max(warning_scores)) if len(warning_scores) > 0 else 0.0
        aggressiveness = max(0.5, 1.0 - 0.25 * max_warning)
        
        options = []
        minlead = {}
        for r in self.routes:
            i, e, k, lane, path, cps, tail, head = r
            if not o['action_mask'][i]:
                continue
            cap = min(float(u[p]) for p in path)
            lead = sum(max(0, int(tau[p])) for p in path)
            closure = 0.0
            war_penalty = 0.0
            min_open = 1.0
            
            for cp in cps:
                ix = self.cp.get(cp)
                if ix is not None:
                    opened = float(open_fractions[ix])
                    min_open = min(min_open, opened)
                    closure += 8 * (1 - opened)
                    pool = self.s['commodities']['pool'][k]
                    field = 'graph_now.kappa.' + str(pool)
                    if field in o:
                        cap = min(cap, float(o[field][ix]) * opened)
                    risk = int(war_risk[ix])
                    if risk > 0:
                        war_penalty += 0.3 * risk
            
            if cap <= 0 or lead >= remaining:
                continue
            pair = (head, k)
            minlead[pair] = min(minlead.get(pair, 1e9), lead + closure)
            expense = sum(float(costs[p]) + float(tariffs[p, k]) * float(values[k]) for p in path)
            score = expense / max(float(values[k]), 1.0) + 0.035 * lead + 0.18 * closure + war_penalty
            options.append((score, r, cap, lead, min_open))
        
        flows = np.zeros(len(self.routes), dtype=float)
        edge_used = np.zeros(len(u))
        targets = {}
        queue_at_cp = {}
        
        if 'queue_lots.qty' in o:
            for key, row in zip(self.l.get('lot_keys', []), o['queue_lots.qty']):
                cp, k, _, _ = key
                queue_at_cp[(cp, k)] = queue_at_cp.get((cp, k), 0) + float(np.sum(row))
        
        for pair in self.dest:
            lead = minlead.get(pair, 0)
            horizon = min(remaining, max(1.0, lead + 2.0))
            target = rates.get(pair, 0) * horizon
            
            if pair in self.demands:
                j = self.demands[pair]
                sink_idx = self.sinks.get(pair)
                n = min(forecast.shape[1], max(1, int(np.ceil(horizon))))
                target = float(np.sum(forecast[j, :n])) + max(0, horizon - n) * rates[pair]
                target += float(o['backlog.qty'][j])
                if sink_idx in self.backlog_sinks:
                    target *= 1.15
            
            targets[pair] = target * aggressiveness
        
        for score, r, cap, lead, min_open in sorted(options, key=lambda x: x[0]):
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
        
        override_qty = np.zeros(len(self.override_slots_list['chokepoint']))
        release_mode = np.zeros(len(self.l['release_pairs']), dtype=np.int64)
        
        for idx, (cp, k, out_edge, lane) in enumerate(zip(
            self.override_slots_list['chokepoint'],
            self.override_slots_list['k'],
            self.override_slots_list['out_edge'],
            self.override_slots_list['lane']
        )):
            if k not in self.tanker_commodities:
                continue
            if 'override_mask' in o and not o['override_mask'][idx]:
                continue
            
            cp_idx = self.cp.get(cp)
            if cp_idx is None:
                continue
            
            opened = float(open_fractions[cp_idx])
            pool = self.s['commodities']['pool'][k]
            kappa = float(o.get('graph_now.kappa.tb', [1.0])[cp_idx]) if pool == 'tb' else float(o.get('graph_now.kappa.ct', [1.0])[cp_idx])
            
            queued = queue_at_cp.get((cp, k), 0.0)
            throughput = kappa * opened
            
            if queued > throughput * 0.7 and opened > 0.5:
                override_qty[idx] = min(queued, throughput * 0.9)
                release_mode_idx = next((i for i, (c, c_k) in enumerate(self.l['release_pairs']) if c == cp and c_k == k), None)
                if release_mode_idx is not None:
                    release_mode[release_mode_idx] = 1
            elif opened < 0.3:
                release_mode_idx = next((i for i, (c, c_k) in enumerate(self.l['release_pairs']) if c == cp and c_k == k), None)
                if release_mode_idx is not None:
                    release_mode[release_mode_idx] = 2
        
        return {'flows': flows, 'override_qty': override_qty, 'release_mode': release_mode}