# 0.5326952411612738
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
        
        # Backlog-first strategy: prioritize serving backlog
        backlog_load = np.zeros(n)
        for i in range(n):
            edge_head = self.edge_head[self.slot_edge[i]]
            k = self.slot_k[i]
            key = (int(edge_head), int(k))
            if key in self.demand_idx:
                backlog_load[i] = backlog[self.demand_idx[key]]
        
        # Group by source
        groups = {}
        for i in range(n):
            key = (int(self.edge_tail[self.slot_edge[i]]), int(self.slot_k[i]))
            if key in self.stock_idx:
                groups.setdefault(key, []).append(i)
        
        # Allocate within each source: backlog first, then forecast
        for key, idxs in groups.items():
            avail = stock[self.stock_idx[key]]
            remaining = avail
            
            # Phase 1: allocate to backlog
            idxs_backlog_sorted = sorted(idxs, key=lambda i: -backlog_load[i])
            for i in idxs_backlog_sorted:
                if backlog_load[i] > 0:
                    allocation = min(slot_cap[i], backlog_load[i], remaining)
                    flows[i] = allocation
                    remaining -= allocation
            
            # Phase 2: allocate remainder to forecast
            if remaining > 0:
                forecast_load = np.zeros(len(idxs))
                for idx, i in enumerate(idxs):
                    if flows[i] == 0:  # Not already allocated
                        edge_head = self.edge_head[self.slot_edge[i]]
                        k = self.slot_k[i]
                        key_d = (int(edge_head), int(k))
                        if key_d in self.demand_idx:
                            d_idx = self.demand_idx[key_d]
                            forecast_load[idx] = demand_forecast[d_idx, 0] if demand_forecast.shape[1] > 0 else 0.0
                
                idxs_forecast_sorted = sorted(enumerate(idxs), key=lambda x: -forecast_load[x[0]])
                for _, i in idxs_forecast_sorted:
                    if flows[i] == 0:
                        allocation = min(slot_cap[i], remaining)
                        flows[i] = allocation
                        remaining -= allocation
        
        flows = flows * mask
        
        override_qty = np.zeros(self.n_over)
        release_mode = np.zeros(self.n_rel, dtype=np.int64)
        
        war_risk = np.array(obs["graph_now.war_risk"], dtype=int) if "graph_now.war_risk" in obs else np.zeros(len(self.chokepoints), dtype=int)
        for j in range(self.n_rel):
            if j < len(self.chokepoints):
                if war_risk[j] > 0:
                    release_mode[j] = 2  # hold at war-risk chokepoints
        
        return {"flows": flows, "override_qty": override_qty, "release_mode": release_mode}