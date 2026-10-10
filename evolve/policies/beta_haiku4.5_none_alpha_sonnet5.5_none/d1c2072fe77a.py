# 0.5424119039731871
import numpy as np

class Agent:
    def __init__(self, config=None):
        st = config['static']
        self.n_over = len(st['override_slots']['chokepoint'])
        self.n_rel = len(config['layout']['release_pairs'])
        ed = st['edges']
        self.u0 = np.array([np.inf if x is None else float(x) for x in ed['u0']], dtype=float)
        self.se = [int(x) for x in st['action_slots']['edge']]
        self.sk = [int(x) for x in st['action_slots']['k']]
        self.sl = list(st['action_slots']['lane'])
        self.le = st['lanes']['edges']
        self.head = [int(x) for x in ed['head']]
        self.tail = [int(x) for x in ed['tail']]
        self.stock_idx = {(int(a), int(b)): r for r, (a, b) in enumerate(config['layout']['stock_slots'])}
        self.demand_idx = {(int(a), int(b)): r for r, (a, b) in enumerate(config['layout']['demands'])}
        self.sinks_pi = np.array([float(x) for x in st['sinks']['pi']], dtype=float)
        self.chokepoints = [int(x) for x in st['chokepoints']] if 'chokepoints' in st else []

    def act(self, obs):
        u = np.array(obs['graph_now.u'], dtype=float)
        uo = np.array(obs['graph_now.u.observed']) if 'graph_now.u.observed' in obs else np.ones_like(u)
        cap = np.where(uo > 0, u, self.u0)
        cap = np.clip(np.where(np.isfinite(cap), cap, 0.0), 0, None)
        
        c = np.array(obs['graph_now.c'], dtype=float)
        tariff = np.array(obs['graph_now.tariff'], dtype=float)
        war_risk = np.array(obs['graph_now.war_risk'], dtype=int) if 'graph_now.war_risk' in obs else np.zeros(len(self.chokepoints), dtype=int)
        
        stock = np.clip(np.array(obs['stock.qty'], dtype=float), 0, None)
        backlog = np.clip(np.array(obs['backlog.qty'], dtype=float), 0, None)
        fc = np.clip(np.array(obs['demand_forecast.qty'], dtype=float), 0, None)
        mask = np.array(obs['action_mask'], dtype=float)
        
        n = len(self.se)
        
        slot_cap = np.zeros(n)
        for i in range(n):
            lane = self.sl[i]
            if lane is None or lane >= len(self.le) or not self.le[lane]:
                cs = cap[self.se[i]]
            else:
                cs = min(cap[int(e)] for e in self.le[lane])
            slot_cap[i] = max(cs, 0.0)
        
        # High backlog weight + strong penalty multiplier
        demand_pressure = np.zeros(n)
        for i in range(n):
            key = (self.head[self.se[i]], self.sk[i])
            if key in self.demand_idx:
                d = self.demand_idx[key]
                press = backlog[d] * 5.0  # High backlog emphasis
                for h in range(min(6, fc.shape[1])):
                    w = max(0.5, 1.0 - 0.12 * h)
                    press += fc[d, h] * w
                pi = self.sinks_pi[d] if d < len(self.sinks_pi) else 1000.0
                press *= (1.0 + pi / 1000.0)  # Strong penalty scaling
                demand_pressure[i] = press
        
        slot_cost = np.zeros(n)
        for i in range(n):
            cost_val = c[self.se[i]] + tariff[self.se[i], self.sk[i]]
            slot_cost[i] = max(cost_val, 0.0)
        
        groups = {}
        for i in range(n):
            key = (self.tail[self.se[i]], self.sk[i])
            if key in self.stock_idx:
                groups.setdefault(key, []).append(i)
        
        flows = np.zeros(n)
        for key, idxs in groups.items():
            rem = stock[self.stock_idx[key]]
            if rem > 0 and idxs:
                for i in sorted(idxs, key=lambda i: (-demand_pressure[i], slot_cost[i])):
                    if mask[i] <= 0:
                        continue
                    q = min(slot_cap[i], rem)
                    if q > 0:
                        flows[i] = q
                        rem -= q
        
        override_qty = np.zeros(self.n_over)
        release_mode = np.zeros(self.n_rel, dtype=np.int64)
        for j in range(self.n_rel):
            if j < len(self.chokepoints):
                chp = self.chokepoints[j]
                if chp < len(war_risk) and war_risk[chp] > 0:
                    release_mode[j] = 2
        
        return {'flows': flows, 'override_qty': override_qty, 'release_mode': release_mode}