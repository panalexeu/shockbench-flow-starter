# 0.46686011933475713
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
        
        # Override and release info
        self.override_chk = [int(x) for x in st['override_slots']['chokepoint']]
        self.override_k = [int(x) for x in st['override_slots']['k']]
        self.release_chk = [int(config['layout']['release_pairs'][j][0]) if j < len(config['layout']['release_pairs']) else -1 for j in range(self.n_rel)]
        
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
        war_risk = np.array(obs['graph_now.war_risk'], dtype=int) if 'graph_now.war_risk' in obs else np.zeros(len(self.chk_nodes), dtype=int)
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

        # Slot capacities and chokepoint status
        slot_cap = np.zeros(n)
        slot_chk = np.full(n, -1, dtype=int)  # Which chokepoint does this slot use? -1 if none
        
        for i in range(n):
            lane = self.sl[i]
            if lane is None or lane >= len(self.le) or not self.le[lane]:
                cs = cap[self.se[i]]
                if lane is None:
                    slot_chk[i] = -1
            else:
                cs = min(cap[int(e)] for e in self.le[lane])
                if lane < len(self.lc) and self.lc[lane]:
                    slot_chk[i] = int(self.lc[lane][0])  # Use first chokepoint
                    for ch in self.lc[lane]:
                        p = self.chk_pos.get(int(ch))
                        if p is not None and p < len(opn) and opn[p] < 0.01:
                            cs = 0.0
                            break
            slot_cap[i] = max(cs, 0.0)

        # Slot priority metrics
        slot_pressure = np.zeros(n)
        slot_cost = np.zeros(n)
        
        for i in range(n):
            slot_cost[i] = c[self.se[i]] + tariff[self.se[i], self.sk[i]]
            
            key = (self.slot_head[i], self.sk[i])
            if key in self.demand_idx:
                d_idx = self.demand_idx[key]
                p = backlog[d_idx] * 3.5  # Strong backlog priority
                for h in range(min(5, fc.shape[1])):
                    p += fc[d_idx, h] * np.exp(-0.12 * h)
                have = intransit.get(key, 0.0) * 0.65
                slot_pressure[i] = max(p - have, 0.0)
            else:
                slot_pressure[i] = 0.0

        # Group by source
        groups = {}
        for i in range(n):
            key = (self.tail[self.se[i]], self.sk[i])
            if key in self.stock_idx:
                groups.setdefault(key, []).append(i)

        flows = np.zeros(n)
        for key, idxs in groups.items():
            rem = stock[self.stock_idx[key]]
            for i in sorted(idxs, key=lambda j: (-slot_pressure[j], slot_cost[j])):
                if mask[i] <= 0 or rem < 1e-9:
                    continue
                q = min(slot_cap[i], rem)
                flows[i] = q
                rem -= q

        flows = flows * mask
        
        # Tanker release strategy: hold at high-risk chokepoints
        override_qty = np.zeros(self.n_over)
        release_mode = np.zeros(self.n_rel, dtype=np.int64)
        
        for j in range(self.n_rel):
            chk_idx = self.chk_pos.get(self.release_chk[j], -1)
            if chk_idx >= 0 and chk_idx < len(war_risk):
                if war_risk[chk_idx] > 0:  # Any war risk
                    release_mode[j] = 2  # Hold cargo
                elif opn[chk_idx] < 0.5:  # Significantly constrained
                    release_mode[j] = 2  # Hold cargo
                else:
                    release_mode[j] = 0  # Default release
            else:
                release_mode[j] = 0
        
        return {'flows': flows, 'override_qty': override_qty, 'release_mode': release_mode}