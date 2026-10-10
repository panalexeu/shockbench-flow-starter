# 0.46752930300149365
import numpy as np

class Agent:
    def __init__(self, config=None):
        self.static = config["static"]
        st = self.static
        self.edges = np.array(st["action_slots"]["edge"], dtype=int)
        self.lanes = st["action_slots"]["lane"]
        self.cp_nodes = list(config["layout"]["chokepoints"])
        self.cp_index = {}
        for i, n in enumerate(self.cp_nodes):
            n = n[0] if isinstance(n, (list, tuple)) else n
            self.cp_index[int(n)] = i
        self.lane_cps = st["lanes"]["chokepoints"]
        aspace = config["spaces"]["action"]
        self.override_qty = np.zeros(aspace["override_qty"]["shape"])
        self.release_mode = np.zeros(aspace["release_mode"]["shape"], dtype=np.int64)

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
        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}
