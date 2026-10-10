# 0.5569052383364809
import numpy as np

class Agent:
    THR = 0.1
    SORT = False

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
        self.lc = st['lanes']['chokepoints']
        self.cpi = {int(nd): r for r, nd in enumerate(config['layout']['chokepoints'])}
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
        op = np.array(obs['graph_now.open'], dtype=float)
        stock = np.clip(np.array(obs['stock.qty'], dtype=float), 0, None)
        mask = np.array(obs['action_mask'], dtype=float)
        n = len(self.se)
        flows = np.zeros(n)
        for i in range(n):
            lane = self.sl[i]
            if lane is None:
                c = cap[self.se[i]]
            else:
                es = self.le[lane]
                c = min(cap[e] for e in es) if es else 0.0
                for nd in self.lc[lane]:
                    j = self.cpi.get(int(nd))
                    if j is not None and op[j] < self.THR:
                        c = 0.0
            flows[i] = max(c, 0.0)
        flows = flows * mask
        for key, idxs in self.groups.items():
            avail = stock[self.stock_idx[key]]
            tot = sum(flows[i] for i in idxs)
            if tot > avail and tot > 0:
                s = avail / tot
                for i in idxs:
                    flows[i] *= s
        return {'flows': flows, 'override_qty': np.zeros(self.n_over), 'release_mode': np.zeros(self.n_rel, dtype=np.int64)}
