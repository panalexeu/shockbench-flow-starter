# 0.5772052561350034
import numpy as np

class Agent:
    def __init__(self, config=None):
        st = config["static"]
        self.st = st
        self.config = config
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

    def act(self, obs):
        # Simpler approach: full capacity, demand-weighted allocation within source group
        u = np.array(obs["graph_now.u"], dtype=float)
        uo = np.array(obs["graph_now.u.observed"]) if "graph_now.u.observed" in obs else np.ones_like(u)
        cap = np.where(uo > 0, u, self.u0)
        cap = np.where(np.isfinite(cap), cap, 0.0)
        
        stock = np.array(obs["stock.qty"], dtype=float)
        stock = np.clip(stock, 0, None)
        backlog = np.array(obs["backlog.qty"], dtype=float)
        backlog = np.clip(backlog, 0, None)
        demand_forecast = np.array(obs["demand_forecast.qty"], dtype=float)
        demand_forecast = np.clip(demand_forecast, 0, None)
        
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
        
        # Backlog only (immediate demand)
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
        
        # Allocate proportionally to backlog within group
        for key, idxs in groups.items():
            avail = max(stock[self.stock_idx[key]], 0.0)
            
            total_backlog = sum(backlog_load[i] for i in idxs)
            
            if total_backlog > 0:
                # Proportional allocation by backlog
                for i in idxs:
                    fraction = backlog_load[i] / total_backlog
                    flows[i] = min(fraction * avail, slot_cap[i])
            else:
                # No backlog: distribute equally or by forecast
                total_forecast = sum(demand_forecast[self.demand_idx[(int(self.edge_head[self.slot_edge[i]]), int(self.slot_k[i]))], 0] if (int(self.edge_head[self.slot_edge[i]]), int(self.slot_k[i])) in self.demand_idx else 0 for i in idxs)
                if total_forecast > 0:
                    for i in idxs:
                        if (int(self.edge_head[self.slot_edge[i]]), int(self.slot_k[i])) in self.demand_idx:
                            f = demand_forecast[self.demand_idx[(int(self.edge_head[self.slot_edge[i]]), int(self.slot_k[i]))], 0]
                            fraction = f / total_forecast
                            flows[i] = min(fraction * avail, slot_cap[i])
                else:
                    # Equal split
                    per_route = avail / len(idxs) if idxs else 0
                    for i in idxs:
                        flows[i] = min(per_route, slot_cap[i])
        
        flows = flows * mask
        
        return {"flows": flows, "override_qty": np.zeros(self.n_over), "release_mode": np.zeros(self.n_rel, dtype=np.int64)}