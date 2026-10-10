# 0.5585669891800147
import numpy as np

class Agent:
    def __init__(self, config=None):
        st = config["static"]
        self.st = st
        self.n_over = len(st["override_slots"]["chokepoint"])
        self.n_rel = len(config["layout"]["release_pairs"])
        edges = st["edges"]
        self.tail = list(edges["tail"])
        self.u0 = np.array([np.inf if x is None else float(x) for x in edges["u0"]], dtype=float)
        self.slot_edge = list(st["action_slots"]["edge"])
        self.slot_k = list(st["action_slots"]["k"])
        self.slot_lane = list(st["action_slots"]["lane"])
        self.lane_edges = st["lanes"]["edges"]
        self.stock_idx = {}
        for r, (nd, k) in enumerate(config["layout"]["stock_slots"]):
            self.stock_idx[(int(nd), int(k))] = r

    def act(self, obs):
        u = np.array(obs["graph_now.u"], dtype=float)
        uo = np.array(obs["graph_now.u.observed"]) if "graph_now.u.observed" in obs else np.ones_like(u)
        cap = np.where(uo > 0, u, self.u0)
        cap = np.where(np.isfinite(cap), cap, 0.0)
        stock = np.array(obs["stock.qty"], dtype=float)
        n = len(self.slot_edge)
        flows = np.zeros(n)
        for i in range(n):
            lane = self.slot_lane[i]
            if lane is None:
                c = cap[self.slot_edge[i]]
            else:
                c = min(cap[e] for e in self.lane_edges[lane])
            flows[i] = max(c, 0.0)
        mask = np.array(obs["action_mask"], dtype=float)
        flows = flows * mask
        groups = {}
        for i in range(n):
            key = (int(self.tail[self.slot_edge[i]]), int(self.slot_k[i]))
            if key in self.stock_idx:
                groups.setdefault(key, []).append(i)
        for key, idxs in groups.items():
            avail = max(stock[self.stock_idx[key]], 0.0)
            tot = sum(flows[i] for i in idxs)
            if tot > avail and tot > 0:
                s = avail / tot
                for i in idxs:
                    flows[i] *= s
        return {"flows": flows, "override_qty": np.zeros(self.n_over), "release_mode": np.zeros(self.n_rel, dtype=np.int64)}
