# 0.5326952411612738
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
        
        # Build stock index: (node, k) -> stock index
        self.stock_idx = {}
        for r, (nd, k) in enumerate(config["layout"]["stock_slots"]):
            self.stock_idx[(int(nd), int(k))] = r
        
        # Build demand index: (sink_node, k) -> demand index
        self.demand_idx = {}
        for r, (nd, k) in enumerate(config["layout"]["demands"]):
            self.demand_idx[(int(nd), int(k))] = r

    def act(self, obs):
        u = np.array(obs["graph_now.u"], dtype=float)
        uo = np.array(obs["graph_now.u.observed"]) if "graph_now.u.observed" in obs else np.ones_like(u)
        cap = np.where(uo > 0, u, self.u0)
        cap = np.where(np.isfinite(cap), cap, 0.0)
        
        stock = np.array(obs["stock.qty"], dtype=float)
        backlog = np.array(obs["backlog.qty"], dtype=float)
        demand_forecast = np.array(obs["demand_forecast.qty"], dtype=float)
        
        mask = np.array(obs["action_mask"], dtype=float)
        
        n = len(self.slot_edge)
        flows = np.zeros(n)
        
        # Calculate capacity per slot (accounting for lanes)
        slot_cap = np.zeros(n)
        for i in range(n):
            lane = self.slot_lane[i]
            if lane is None:
                c = cap[self.slot_edge[i]]
            else:
                c = min(cap[e] for e in self.lane_edges[lane]) if self.lane_edges[lane] else 0.0
            slot_cap[i] = max(c, 0.0)
        
        # Group slots by source (tail node, commodity)
        groups = {}
        for i in range(n):
            key = (int(self.edge_tail[self.slot_edge[i]]), int(self.slot_k[i]))
            if key in self.stock_idx:
                groups.setdefault(key, []).append(i)
        
        # Priority scoring: prioritize demand with higher immediate shortfalls
        # Look at current backlog + immediate demand (week 0 forecast)
        demand_pressure = np.zeros(n)
        for i in range(n):
            edge_head = self.edge_head[self.slot_edge[i]]
            k = self.slot_k[i]
            key = (int(edge_head), int(k))
            if key in self.demand_idx:
                d_idx = self.demand_idx[key]
                # Combine backlog and forecasted demand
                press = backlog[d_idx] + max(demand_forecast[d_idx, 0], 0.0)
                demand_pressure[i] = press
        
        # Allocate based on demand pressure and capacity
        for key, idxs in groups.items():
            avail = max(stock[self.stock_idx[key]], 0.0)
            # Sort indices by demand pressure (descending)
            idxs_sorted = sorted(idxs, key=lambda i: -demand_pressure[i])
            
            remaining = avail
            for i in idxs_sorted:
                slot_request = min(slot_cap[i], remaining)
                flows[i] = slot_request
                remaining -= slot_request
        
        flows = flows * mask
        
        return {"flows": flows, "override_qty": np.zeros(self.n_over), "release_mode": np.zeros(self.n_rel, dtype=np.int64)}