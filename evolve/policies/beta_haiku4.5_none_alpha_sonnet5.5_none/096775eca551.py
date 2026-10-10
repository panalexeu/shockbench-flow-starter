# 0.5530793899094667
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
        self.v = np.array([float(x) for x in st['commodities']['v']], dtype=float)
        self.stock_idx = {(int(a), int(b)): r for r, (a, b) in enumerate(config['layout']['stock_slots'])}
        self.demand_idx = {(int(a), int(b)): r for r, (a, b) in enumerate(config['layout']['demands'])}
        self.sinks_pi = np.array([float(x) for x in st['sinks']['pi']], dtype=float)
        self.chk_nodes = [int(x) for x in config['layout']['chokepoints']]
        self.chk_pos = {n: i for i, n in enumerate(self.chk_nodes)}
        self.lc = st['lanes']['chokepoints']
        
        self.slot_head = []
        for i in range(len(self.se)):
            lane = self.sl[i]
            if lane is None or lane >= len(self.le) or not self.le[lane]:
                self.slot_head.append(self.head[self.se[i]])
            else:
                self.slot_head.append(self.head[int(self.le[lane][-1])])

    def act(self, obs):
        u = np.array(obs['graph_now.u'], dtype=float)
        uo = np.array(obs['graph_now.u.observed']) if 'graph_now.u.observed' in obs else np.ones_like(u)
        cap = np.where(uo > 0, u, self.u0)
        cap = np.clip(np.where(np.isfinite(cap), cap, 0.0), 0, None)
        c = np.array(obs['graph_now.c'], dtype=float)
        tariff = np.array(obs['graph_now.tariff'], dtype=float)
        opn = np.array(obs['graph_now.open'], dtype=float)
        stock = np.clip(np.array(obs['stock.qty'], dtype=float), 0, None)
        backlog = np.clip(np.array(obs['backlog.qty'], dtype=float), 0, None)
        fc = np.clip(np.array(obs['demand_forecast.qty'], dtype=float), 0, None)
        mask = np.array(obs['action_mask'], dtype=float)
        n = len(self.se)

        # Pipeline inventory
        intransit = {}
        pq = np.array(obs['pipeline.qty'], dtype=float)
        pe = np.array(obs['pipeline.edge'], dtype=int)
        pk = np.array(obs['pipeline.k'], dtype=int)
        po = np.array(obs['pipeline.qty.observed']) if 'pipeline.qty.observed' in obs else np.ones_like(pq)
        for i in range(len(pq)):
            if po[i] > 0 and 0 <= pe[i] < len(self.head):
                key = (self.head[pe[i]], int(pk[i]))
                intransit[key] = intransit.get(key, 0.0) + pq[i]

        # Slot capacities
        slot_cap = np.zeros(n)
        for i in range(n):
            lane = self.sl[i]
            if lane is None or lane >= len(self.le) or not self.le[lane]:
                cs = cap[self.se[i]]
            else:
                cs = min(cap[int(e)] for e in self.le[lane])
                if lane < len(self.lc):
                    for ch in self.lc[lane]:
                        p = self.chk_pos.get(int(ch))
                        if p is not None and p < len(opn) and opn[p] < 0.01:
                            cs = 0.0
                            break
            slot_cap[i] = max(cs, 0.0)

        # Cost-benefit: shortage penalty vs. freight+tariff cost
        slot_benefit = np.zeros(n)
        slot_cost = np.zeros(n)
        
        for i in range(n):
            # Direct freight cost
            cost_val = c[self.se[i]] + tariff[self.se[i], self.sk[i]]
            slot_cost[i] = cost_val
            
            key = (self.slot_head[i], self.sk[i])
            if key in self.demand_idx:
                d_idx = self.demand_idx[key]
                # Shortage penalty per unit
                pi = self.sinks_pi[d_idx]
                # Total demand from backlog + near-term forecast
                bl = backlog[d_idx]
                fcst = sum(fc[d_idx, h] for h in range(min(4, fc.shape[1])))
                # How much can this slot contribute to meeting demand?
                # Benefit = shortage_penalty * min(slot_cap, unmet_demand)
                # But we need to estimate unmet demand considering pipeline
                available = intransit.get(key, 0.0) * 0.6  # conservative pipeline credit
                unmet = max(bl + fcst - available, 0.0)
                # Benefit-to-cost ratio
                slot_benefit[i] = pi * min(slot_cap[i], unmet)
            else:
                slot_benefit[i] = 0.0

        # Sort by benefit/cost ratio
        slot_score = np.zeros(n)
        for i in range(n):
            if slot_cap[i] > 1e-9:
                slot_score[i] = slot_benefit[i] / (1.0 + slot_cost[i])

        # Group by source
        groups = {}
        for i in range(n):
            key = (self.tail[self.se[i]], self.sk[i])
            if key in self.stock_idx:
                groups.setdefault(key, []).append(i)

        flows = np.zeros(n)
        for key, idxs in groups.items():
            rem = stock[self.stock_idx[key]]
            # Greedy: highest benefit-to-cost first
            for i in sorted(idxs, key=lambda j: -slot_score[j]):
                if mask[i] <= 0 or rem < 1e-9:
                    continue
                q = min(slot_cap[i], rem)
                flows[i] = q
                rem -= q

        flows = flows * mask
        return {'flows': flows, 'override_qty': np.zeros(self.n_over), 'release_mode': np.zeros(self.n_rel, dtype=np.int64)}