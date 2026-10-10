# 0.5228902893368009
import numpy as np

class Agent:
    F = 1.4
    BL = 2.0
    H = 3
    GREEDY = True

    def __init__(self, config=None):
        st = config['static']
        self.edges = np.array(st['action_slots']['edge'], dtype=int)
        self.ks = [int(k) for k in st['action_slots']['k']]
        self.lanes = st['action_slots']['lane']
        tails = st['edges']['tail']
        heads = st['edges']['head']
        self.v = [float(x) for x in st['commodities']['v']]
        self.slot_tail = [int(tails[e]) for e in self.edges]
        self.slot_head = []
        self.slot_edges = []
        for s, e in enumerate(self.edges):
            ln = self.lanes[s]
            se = [int(e)]
            if ln is not None:
                se = [int(x) for x in st['lanes']['edges'][int(ln)]]
            self.slot_edges.append(se)
            self.slot_head.append(int(heads[se[-1]]))
        lay = config['layout']
        self.stock_index = {(int(a), int(b)): i for i, (a, b) in enumerate(lay['stock_slots'])}
        self.supply_index = {(int(a), int(b)): i for i, (a, b) in enumerate(lay['supply_slots'])}
        self.dem_index = {(int(a), int(b)): i for i, (a, b) in enumerate(lay['demands'])}
        self.cp_index = {}
        for i, n in enumerate(lay['chokepoints']):
            n = n[0] if isinstance(n, (list, tuple)) else n
            self.cp_index[int(n)] = i
        self.lane_cps = st['lanes']['chokepoints']
        aspace = config['spaces']['action']
        self.override_qty = np.zeros(aspace['override_qty']['shape'])
        self.release_mode = np.zeros(aspace['release_mode']['shape'], dtype=np.int64)
        self.groups = {}
        self.sink_groups = {}
        for s in range(len(self.edges)):
            self.groups.setdefault((self.slot_tail[s], self.ks[s]), []).append(s)
            key = (self.slot_head[s], self.ks[s])
            if key in self.dem_index:
                self.sink_groups.setdefault(key, []).append(s)

    def act(self, observation):
        u = np.nan_to_num(np.asarray(observation['graph_now.u'], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        flows = u[self.edges].copy()
        mask = np.asarray(observation['action_mask'], dtype=float)
        if int(np.asarray(observation['action_mask.observed']).ravel()[0]) == 0:
            mask = np.ones_like(mask)
        flows = flows * mask
        op = np.asarray(observation['graph_now.open'], dtype=float)
        for s, lane in enumerate(self.lanes):
            if lane is None:
                continue
            for c in self.lane_cps[int(lane)]:
                i = self.cp_index.get(int(c))
                if i is not None and op[i] < 0.05:
                    flows[s] = 0.0
                    break
        flows = np.maximum(flows, 0.0)
        stock = np.asarray(observation['stock.qty'], dtype=float)
        stock_obs = np.asarray(observation['stock.qty.observed']).astype(bool)
        sup = np.asarray(observation['graph_now.supply.avail'], dtype=float)
        sup_obs = np.asarray(observation['graph_now.supply.avail.observed']).astype(bool)
        for key, slots in self.groups.items():
            avail = 0.0
            known = False
            si = self.stock_index.get(key)
            if si is not None and stock_obs[si]:
                avail += max(stock[si], 0.0)
                known = True
            pi = self.supply_index.get(key)
            if pi is not None and sup_obs[pi]:
                avail += max(sup[pi], 0.0)
                known = True
            if not known:
                continue
            req = float(sum(flows[s] for s in slots))
            if req > avail and req > 0:
                sc = avail / req
                for s in slots:
                    flows[s] *= sc
        try:
            fc = np.asarray(observation['demand_forecast.qty'], dtype=float)
            fo = np.asarray(observation['demand_forecast.qty.observed']).astype(bool)
            bl = np.asarray(observation['backlog.qty'], dtype=float)
            bo = np.asarray(observation['backlog.qty.observed']).astype(bool)
            cc = np.nan_to_num(np.asarray(observation['graph_now.c'], dtype=float))
            tf = np.nan_to_num(np.asarray(observation['graph_now.tariff'], dtype=float))
            for key, slots in self.sink_groups.items():
                row = self.dem_index[key]
                m = fo[row, :self.H]
                if not m.any():
                    continue
                cap = self.F * float(np.mean(fc[row, :self.H][m]))
                if bo[row]:
                    cap += self.BL * max(bl[row], 0.0)
                cap = max(cap, 0.0)
                req = float(sum(flows[s] for s in slots))
                if req > cap and req > 0:
                    if self.GREEDY and len(slots) > 1:
                        k = key[1]
                        def cost(s):
                            return sum(cc[e] + tf[e, k] * self.v[k] for e in self.slot_edges[s])
                        rem = cap
                        for s in sorted(slots, key=cost):
                            take = min(flows[s], rem)
                            flows[s] = take
                            rem -= take
                    else:
                        sc = cap / req
                        for s in slots:
                            flows[s] *= sc
        except Exception:
            pass
        return {'flows': flows, 'override_qty': self.override_qty, 'release_mode': self.release_mode}
