# 0.5605881713189297
import numpy as np

class Agent:
    MARGIN = 1.25
    H = 5
    TV = 1.0

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
        self.lc = st['lanes']['chokepoints']
        self.head = [int(x) for x in ed['head']]
        self.tail = [int(x) for x in ed['tail']]
        self.v = np.array([float(x) for x in st['commodities']['v']], dtype=float)
        self.chk_pos = {int(n): i for i, n in enumerate(config['layout']['chokepoints'])}
        self.stock_idx = {(int(a), int(b)): r for r, (a, b) in enumerate(config['layout']['stock_slots'])}
        self.demand_idx = {(int(a), int(b)): r for r, (a, b) in enumerate(config['layout']['demands'])}
        n = len(self.se)
        self.slot_head = []
        for i in range(n):
            lane = self.sl[i]
            if lane is None or lane >= len(self.le) or not self.le[lane]:
                self.slot_head.append(self.head[self.se[i]])
            else:
                self.slot_head.append(self.head[int(self.le[lane][-1])])

    def _fallback(self, obs):
        u = np.array(obs['graph_now.u'], dtype=float)
        cap = np.clip(np.where(np.isfinite(u), u, 0.0), 0, None)
        n = len(self.se)
        flows = np.zeros(n)
        for i in range(n):
            lane = self.sl[i]
            if lane is None:
                flows[i] = cap[self.se[i]]
            else:
                flows[i] = min(cap[int(e)] for e in self.le[lane])
        flows = flows * np.array(obs['action_mask'], dtype=float)
        return {'flows': flows, 'override_qty': np.zeros(self.n_over), 'release_mode': np.zeros(self.n_rel, dtype=np.int64)}

    def act(self, obs):
        try:
            return self._act(obs)
        except Exception:
            return self._fallback(obs)

    def _act(self, obs):
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

        intransit = {}
        pq = np.array(obs['pipeline.qty'], dtype=float)
        pe = np.array(obs['pipeline.edge'], dtype=int)
        pk = np.array(obs['pipeline.k'], dtype=int)
        po = np.array(obs['pipeline.qty.observed'])
        for j in range(len(pq)):
            if po[j] > 0 and 0 <= pe[j] < len(self.head):
                key = (self.head[pe[j]], int(pk[j]))
                intransit[key] = intransit.get(key, 0.0) + pq[j]

        slot_cap = np.zeros(n)
        for i in range(n):
            lane = self.sl[i]
            if lane is None or lane >= len(self.le) or not self.le[lane]:
                cs = cap[self.se[i]]
            else:
                cs = min(cap[int(e)] for e in self.le[lane])
                for ch in self.lc[lane]:
                    p = self.chk_pos.get(int(ch))
                    if p is not None and p < len(opn) and opn[p] < 0.05:
                        cs = 0.0
            slot_cap[i] = max(cs, 0.0)

        need = {}
        press = np.zeros(n)
        for i in range(n):
            key = (self.slot_head[i], self.sk[i])
            if key in self.demand_idx:
                d = self.demand_idx[key]
                p = backlog[d] * 2.5
                tot = backlog[d]
                for h in range(min(self.H, fc.shape[1])):
                    p += fc[d, h] * np.exp(-0.1 * h)
                    tot += fc[d, h]
                press[i] = p
                have = intransit.get(key, 0.0)
                if key in self.stock_idx:
                    have += stock[self.stock_idx[key]]
                need[key] = max(tot * self.MARGIN - have, 0.0)

        slot_cost = np.array([max(c[self.se[i]] + self.TV * tariff[self.se[i], self.sk[i]] * self.v[self.sk[i]], 0.0) for i in range(n)])

        groups = {}
        for i in range(n):
            key = (self.tail[self.se[i]], self.sk[i])
            if key in self.stock_idx:
                groups.setdefault(key, []).append(i)

        flows = np.zeros(n)
        for key, idxs in groups.items():
            rem = stock[self.stock_idx[key]]
            for i in sorted(idxs, key=lambda i: (-press[i], slot_cost[i])):
                if mask[i] <= 0:
                    continue
                q = min(slot_cap[i], rem)
                sk = (self.slot_head[i], self.sk[i])
                if sk in need:
                    q = min(q, need[sk])
                if q > 0:
                    flows[i] = q
                    rem -= q
                    if sk in need:
                        need[sk] -= q
        return {'flows': flows, 'override_qty': np.zeros(self.n_over), 'release_mode': np.zeros(self.n_rel, dtype=np.int64)}
