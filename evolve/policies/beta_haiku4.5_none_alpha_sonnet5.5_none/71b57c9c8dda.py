# 0.5361672171480927
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
        self.c0 = np.array([float(x) if x is not None else 0.0 for x in edges["c0"]], dtype=float)
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
        
        # For tanker release strategy
        self.chokepoint_nodes = [int(x) for x in st["chokepoints"]] if "chokepoints" in st else []
        self.release_pairs = config["layout"]["release_pairs"] if "release_pairs" in config["layout"] else []

    def act(self, obs):
        # Get capacities
        u = np.array(obs["graph_now.u"], dtype=float)
        uo = np.array(obs["graph_now.u.observed"]) if "graph_now.u.observed" in obs else np.ones_like(u)
        cap = np.where(uo > 0, u, self.u0)
        cap = np.where(np.isfinite(cap), cap, 0.0)
        cap = np.clip(cap, 0, None)
        
        # Get costs and tariffs
        c = np.array(obs["graph_now.c"], dtype=float)
        tariff = np.array(obs["graph_now.tariff"], dtype=float)
        open_frac = np.array(obs["graph_now.open"], dtype=float) if "graph_now.open" in obs else np.ones(len(self.chokepoint_nodes))
        
        stock = np.array(obs["stock.qty"], dtype=float)
        stock = np.clip(stock, 0, None)
        backlog = np.array(obs["backlog.qty"], dtype=float)
        backlog = np.clip(backlog, 0, None)
        demand_forecast = np.array(obs["demand_forecast.qty"], dtype=float)
        demand_forecast = np.clip(demand_forecast, 0, None)
        
        mask = np.array(obs["action_mask"], dtype=float)
        
        n = len(self.slot_edge)
        flows = np.zeros(n)
        
        # Compute slot capacities
        slot_cap = np.zeros(n)
        for i in range(n):
            lane = self.slot_lane[i]
            if lane is None or lane >= len(self.lane_edges):
                c_slot = cap[self.slot_edge[i]]
            else:
                lane_edges = self.lane_edges[lane]
                if lane_edges:
                    c_slot = min(cap[e] for e in lane_edges)
                else:
                    c_slot = 0.0
            slot_cap[i] = max(c_slot, 0.0)
        
        # Compute cost per slot (use for tie-breaking when demand is similar)
        slot_cost = np.zeros(n)
        for i in range(n):
            edge_idx = self.slot_edge[i]
            k = self.slot_k[i]
            cost_val = c[edge_idx] + tariff[edge_idx, k]
            slot_cost[i] = max(cost_val, 0.0)
        
        # Compute demand priority: backlog dominates
        demand_priority = np.zeros(n)
        for i in range(n):
            edge_head = self.edge_head[self.slot_edge[i]]
            k = self.slot_k[i]
            key = (int(edge_head), int(k))
            if key in self.demand_idx:
                d_idx = self.demand_idx[key]
                # Backlog + 3-week forecast, with high weight on backlog
                priority = backlog[d_idx] * 3.0 + sum(demand_forecast[d_idx, h] for h in range(min(3, demand_forecast.shape[1])))
                demand_priority[i] = priority
        
        # Group slots by source
        groups = {}
        for i in range(n):
            key = (int(self.edge_tail[self.slot_edge[i]]), int(self.slot_k[i]))
            if key in self.stock_idx:
                groups.setdefault(key, []).append(i)
        
        # Allocate: sort by (demand priority desc, cost asc)
        for key, idxs in groups.items():
            avail = max(stock[self.stock_idx[key]], 0.0)
            
            if avail > 0:
                idxs_sorted = sorted(idxs, key=lambda i: (-demand_priority[i], slot_cost[i]))
                
                remaining = avail
                for i in idxs_sorted:
                    slot_request = min(slot_cap[i], remaining)
                    flows[i] = slot_request
                    remaining -= slot_request
        
        flows = flows * mask
        
        # Conservative tanker release: default mode, no overrides
        override_qty = np.zeros(self.n_over)
        release_mode = np.zeros(self.n_rel, dtype=np.int64)
        
        return {"flows": flows, "override_qty": override_qty, "release_mode": release_mode}