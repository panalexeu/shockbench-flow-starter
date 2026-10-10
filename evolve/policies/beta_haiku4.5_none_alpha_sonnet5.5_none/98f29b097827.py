# 0.5254150736034322
import numpy as np

class Agent:
    def __init__(self, config=None):
        st = config['static']
        self.n_over = len(st['override_slots']['chokepoint'])
        self.n_rel = len(config['layout']['release_pairs'])
        ed = st['edges']
        self.tail = list(ed['tail'])
        self.u0 = np.array([np.inf if x is None else float(x) for x in ed['u0']], dtype=float)
        self.se = list(st['action_slots']['edge'])
        self.sk = list(st['action_slots']['k'])
        self.sl = list(st['action_slots']['lane'])
        self.le = st['lanes']['edges']
        self.v = [float(x) for x in st['commodities']['v']]
        self.stock_idx = {}
        for r, (nd, k) in enumerate(config['layout']['stock_slots']):
            self.stock_idx[(int(nd), int(k))] = r
        self.groups = {}
        for i in range(len(self.se)):
            key = (int(self.tail[self.se[i]]), int(self.sk[i]))
            if key in self.stock_idx:
                self.groups.setdefault(key, []).append(i)

    def act(self, obs):
        u = np.array(obs['graph_now.u'], dtype=float)
        uo = np.array(obs['graph_now.u.observed']) if 'graph_now.u.observed' in obs else np.ones_like(u)
        cap = np.where(uo > 0, u, self.u0)
        cap = np.where(np.isfinite(cap), cap, 0.0)
        cap = np.clip(cap, 0, None)
        cc = np.array(obs['graph_now.c'], dtype=float)
        tar = np.array(obs['graph_now.tariff'], dtype=float)
        stock = np.clip(np.array(obs['stock.qty'], dtype=float), 0, None)
        mask = np.array(obs['action_mask'], dtype=float)
        n = len(self.se)
        slot_cap = np.zeros(n)
        cost = np.zeros(n)
        for i in range(n):
            lane = self.sl[i]
            k = int(self.sk[i])
            if lane is None:
                es = [self.se[i]]
            else:
                es = list(self.le[lane])
            c = min(cap[e] for e in es) if es else 0.0
            slot_cap[i] = max(c, 0.0) * mask[i]
            cost[i] = sum(cc[e] + tar[e, k] * self.v[k] for e in es)
        flows = np.zeros(n)
        for key, idxs in self.groups.items():
            rem = stock[self.stock_idx[key]]
            for i in sorted(idxs, key=lambda i: cost[i]):
                q = min(slot_cap[i], rem)
                if q > 0:
                    flows[i] = q
                    rem -= q
        return {'flows': flows, 'override_qty': np.zeros(self.n_over), 'release_mode': np.zeros(self.n_rel, dtype=np.int64)}
