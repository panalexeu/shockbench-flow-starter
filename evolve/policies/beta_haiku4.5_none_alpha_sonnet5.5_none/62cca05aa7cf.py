# 0.5607593631069779

import numpy as np

class Agent:
    def __init__(self, config=None):
        st = config["static"]
        self.n_over = len(st["override_slots"]["chokepoint"])
        self.n_rel = len(config["layout"]["release_pairs"])
        edges = st["edges"]
        self.tail = list(edges["tail"])
        self.head = list(edges["head"])
        self.c0 = np.array([0.0 if x is None else float(x) for x in edges["c0"]], dtype=float)
        self.u0 = np.array([np.inf if x is None else float(x) for x in edges["u0"]], dtype=float)
        self.slot_edge = list(st["action_slots"]["edge"])
        self.slot_k = list(st["action_slots"]["k"])
        self.slot_lane = list(st["action_slots"]["lane"])
        self.lane_edges = st["lanes"]["edges"]
        self.stock_idx = {}
        for r, (nd, k) in enumerate(config["layout"]["stock_slots"]):
            self.stock_idx[(int(nd), int(k))] = r
        self.demand_idx = {}
        for r, (nd, k) in enumerate(config["layout"]["demands"]):
            self.demand_idx[(int(nd), int(k))] = r
        self.sinks_pi = np.array([float(x) for x in st["sinks"]["pi"]], dtype=float)
        n = len(self.slot_edge)
        self.slot_last_head = []
        for i in range(n):
            lane = self.slot_lane[i]
            if lane is None or not self.lane_edges[lane]:
                e = self.slot_edge[i]
            else:
                e = self.lane_edges[lane][-1]
            self.slot_last_head.append(int(self.head[e]))

    def act(self, obs):
        u = np.array(obs["graph_now.u"], dtype=float)
        uo = np.array(obs["graph_now.u.observed"]) if "graph_now.u.observed" in obs else np.ones_like(u)
        cap = np.where(uo > 0, u, self.u0)
        cap = np.where(np.isfinite(cap), cap, 0.0)
        cap = np.clip(cap, 0, None)
        stock = np.clip(np.array(obs["stock.qty"], dtype=float), 0, None)
        backlog = np.clip(np.array(obs["backlog.qty"], dtype=float), 0, None)
        fc = np.clip(np.array(obs["demand_forecast.qty"], dtype=float), 0, None)
        mask = np.array(obs["action_mask"], dtype=float)
        n = len(self.slot_edge)
        
        flows = np.zeros(n)
        for i in range(n):
            lane = self.slot_lane[i]
            if lane is None:
                c_slot = cap[self.slot_edge[i]]
            else:
                le = self.lane_edges[lane] if lane < len(self.lane_edges) else []
                c_slot = min(cap[e] for e in le) if le else 0.0
            flows[i] = max(c_slot, 0.0)
        
        # Multi-horizon demand with shortage penalty weighting
        intransit = {}
        if "pipeline.qty" in obs:
            pq = np.array(obs["pipeline.qty"], dtype=float)
            pe = np.array(obs["pipeline.edge"], dtype=int)
            pk = np.array(obs["pipeline.k"], dtype=int)
            po = np.array(obs["pipeline.qty.observed"]) if "pipeline.qty.observed" in obs else np.ones_like(pq)
            for j in range(len(pq)):
                if po[j] > 0 and 0 <= pe[j] < len(self.head):
                    key = (int(self.head[pe[j]]), int(pk[j]))
                    if key in self.demand_idx:
                        intransit[key] = intransit.get(key, 0.0) + pq[j]
        
        # Demand with penalty-weighted priority
        need = {}
        demand_priority = {}
        for key, d in self.demand_idx.items():
            h = min(4, fc.shape[1])
            tot = backlog[d] * 2.0 + fc[d, :h].sum()
            tot = tot * 1.6 - intransit.get(key, 0.0)
            need[key] = max(tot, 0.0)
            demand_priority[key] = self.sinks_pi[d] if d < len(self.sinks_pi) else 1000.0
        
        # Cost-weighted allocation prioritizing high-penalty demand
        cost_order = [(self.c0[self.slot_edge[i]], -demand_priority.get((self.slot_last_head[i], int(self.slot_k[i])), 0), i) 
                      for i in range(n)]
        cost_order.sort()
        order = [i for _, _, i in cost_order]
        
        for i in order:
            key = (self.slot_last_head[i], int(self.slot_k[i]))
            if key in need and flows[i] > 0:
                q = min(flows[i], need[key])
                flows[i] = q
                need[key] -= q
        
        groups = {}
        for i in range(n):
            key = (int(self.tail[self.slot_edge[i]]), int(self.slot_k[i]))
            if key in self.stock_idx:
                groups.setdefault(key, []).append(i)
        
        for key, idxs in groups.items():
            avail = stock[self.stock_idx[key]]
            tot = sum(flows[i] for i in idxs)
            if tot > avail and tot > 0:
                s = avail / tot
                for i in idxs:
                    flows[i] *= s
        
        flows = flows * mask
        return {"flows": flows, "override_qty": np.zeros(self.n_over), "release_mode": np.zeros(self.n_rel, dtype=np.int64)}