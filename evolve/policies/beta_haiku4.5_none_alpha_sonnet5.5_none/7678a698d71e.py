# 0.536984146132662
import numpy as np

class Agent:
    def __init__(self, config=None):
        st = config["static"]
        self.n_over = len(st["override_slots"]["chokepoint"])
        self.n_rel = len(config["layout"]["release_pairs"])
        edges = st["edges"]
        self.u0 = np.array([np.inf if x is None else float(x) for x in edges["u0"]], dtype=float)
        self.slot_edge = list(st["action_slots"]["edge"])
        self.slot_k = list(st["action_slots"]["k"])
        self.slot_lane = list(st["action_slots"]["lane"])
        self.lane_edges = st["lanes"]["edges"]
        self.edge_head = list(edges["head"])
        self.edge_tail = list(edges["tail"])
        
        self.stock_idx = {}
        for r, (nd, k) in enumerate(config["layout"]["stock_slots"]):
            self.stock_idx[(int(nd), int(k))] = r
        
        self.demand_idx = {}
        for r, (nd, k) in enumerate(config["layout"]["demands"]):
            self.demand_idx[(int(nd), int(k))] = r
        
        self.chokepoints = [int(x) for x in st["chokepoints"]] if "chokepoints" in st else []

    def act(self, obs):
        u = np.array(obs["graph_now.u"], dtype=float)
        uo = np.array(obs["graph_now.u.observed"]) if "graph_now.u.observed" in obs else np.ones_like(u)
        cap = np.where(uo > 0, u, self.u0)
        cap = np.where(np.isfinite(cap), cap, 0.0)
        cap = np.clip(cap, 0, None)
        
        stock = np.clip(np.array(obs["stock.qty"], dtype=float), 0, None)
        backlog = np.clip(np.array(obs["backlog.qty"], dtype=float), 0, None)
        demand_forecast = np.clip(np.array(obs["demand_forecast.qty"], dtype=float), 0, None)
        mask = np.array(obs["action_mask"], dtype=float)
        
        # Track in-transit shipments to sinks
        intransit = {}
        if "pipeline.qty" in obs:
            pq = np.array(obs["pipeline.qty"], dtype=float)
            pe = np.array(obs["pipeline.edge"], dtype=int)
            pk = np.array(obs["pipeline.k"], dtype=int)
            po = np.array(obs["pipeline.qty.observed"]) if "pipeline.qty.observed" in obs else np.ones_like(pq)
            for j in range(len(pq)):
                if po[j] > 0 and 0 <= pe[j] < len(self.edge_head):
                    key = (int(self.edge_head[pe[j]]), int(pk[j]))
                    if key in self.demand_idx:
                        intransit[key] = intransit.get(key, 0.0) + pq[j]
        
        n = len(self.slot_edge)
        flows = np.zeros(n)
        
        # Slot capacities
        slot_cap = np.zeros(n)
        for i in range(n):
            lane = self.slot_lane[i]
            if lane is None or lane >= len(self.lane_edges):
                c_slot = cap[self.slot_edge[i]]
            else:
                lane_edges = self.lane_edges[lane]
                c_slot = min((cap[e] for e in lane_edges), default=0.0) if lane_edges else 0.0
            slot_cap[i] = max(c_slot, 0.0)
        
        # Compute need per sink: backlog + forecast - in_transit
        need = {}
        for key, d in self.demand_idx.items():
            h = min(4, demand_forecast.shape[1])  # 4-week horizon
            tot = backlog[d] + demand_forecast[d, :h].sum()
            tot -= intransit.get(key, 0.0)
            need[key] = max(tot, 0.0)
        
        # Priority: allocate to routes serving high need
        need_per_slot = np.zeros(n)
        for i in range(n):
            edge_head = self.edge_head[self.slot_edge[i]]
            k = self.slot_k[i]
            key = (int(edge_head), int(k))
            need_per_slot[i] = need.get(key, 0.0)
        
        # Group by source
        groups = {}
        for i in range(n):
            key = (int(self.edge_tail[self.slot_edge[i]]), int(self.slot_k[i]))
            if key in self.stock_idx:
                groups.setdefault(key, []).append(i)
        
        # Allocate by need within each source group
        for key, idxs in groups.items():
            avail = stock[self.stock_idx[key]]
            idxs_sorted = sorted(idxs, key=lambda i: -need_per_slot[i])
            remaining = avail
            for i in idxs_sorted:
                slot_request = min(slot_cap[i], remaining)
                flows[i] = slot_request
                remaining -= slot_request
        
        flows = flows * mask
        
        # Tanker strategy: hold at war-risk chokepoints
        override_qty = np.zeros(self.n_over)
        release_mode = np.zeros(self.n_rel, dtype=np.int64)
        
        war_risk = np.array(obs["graph_now.war_risk"], dtype=int) if "graph_now.war_risk" in obs else np.zeros(len(self.chokepoints), dtype=int)
        for j in range(self.n_rel):
            if j < len(self.chokepoints):
                if war_risk[j] > 0:
                    release_mode[j] = 2  # hold
        
        return {"flows": flows, "override_qty": override_qty, "release_mode": release_mode}