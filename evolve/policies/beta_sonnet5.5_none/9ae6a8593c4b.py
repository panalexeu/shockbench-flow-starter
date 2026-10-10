# 0.5102145767973163
import numpy as np

class Agent:
    def __init__(self, config=None):
        self.static = config["static"]
        st = self.static
        self.edges = np.array(st["action_slots"]["edge"], dtype=int)
        self.ks = [int(k) for k in st["action_slots"]["k"]]
        self.lanes = st["action_slots"]["lane"]
        tails = st["edges"]["tail"]
        self.slot_tail = [int(tails[e]) for e in self.edges]
        lay = config["layout"]
        self.stock_index = {(int(a), int(b)): i for i, (a, b) in enumerate(lay["stock_slots"])}
        self.supply_index = {(int(a), int(b)): i for i, (a, b) in enumerate(lay["supply_slots"])}
        self.cp_index = {}
        for i, n in enumerate(lay["chokepoints"]):
            n = n[0] if isinstance(n, (list, tuple)) else n
            self.cp_index[int(n)] = i
        self.lane_cps = st["lanes"]["chokepoints"]
        aspace = config["spaces"]["action"]
        self.override_qty = np.zeros(aspace["override_qty"]["shape"])
        self.release_mode = np.zeros(aspace["release_mode"]["shape"], dtype=np.int64)
        self.groups = {}
        for s in range(len(self.edges)):
            self.groups.setdefault((self.slot_tail[s], self.ks[s]), []).append(s)

    def act(self, observation):
        u = np.nan_to_num(np.asarray(observation["graph_now.u"], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        flows = u[self.edges].copy()
        mask = np.asarray(observation["action_mask"], dtype=float)
        if int(np.asarray(observation["action_mask.observed"]).ravel()[0]) == 0:
            mask = np.ones_like(mask)
        flows = flows * mask
        op = np.asarray(observation["graph_now.open"], dtype=float)
        for s, lane in enumerate(self.lanes):
            if lane is None:
                continue
            for c in self.lane_cps[int(lane)]:
                i = self.cp_index.get(int(c))
                if i is not None and op[i] < 0.05:
                    flows[s] = 0.0
                    break
        flows = np.maximum(flows, 0.0)
        stock = np.asarray(observation["stock.qty"], dtype=float)
        stock_obs = np.asarray(observation["stock.qty.observed"]).astype(bool)
        sup = np.asarray(observation["graph_now.supply.avail"], dtype=float)
        sup_obs = np.asarray(observation["graph_now.supply.avail.observed"]).astype(bool)
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
        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}
