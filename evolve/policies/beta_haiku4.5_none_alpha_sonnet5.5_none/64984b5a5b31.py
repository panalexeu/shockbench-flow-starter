# 0.5526917691517688
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
        u = np.array(obs["graph_now.u"], dtype=float)
        uo = np.array(obs["graph_now.u.observed"]) if "graph_now.u.observed" in obs else np.ones_like(u)
        cap = np.where(uo > 0, u, self.u0)
        cap = np.where(np.isfinite(cap), cap, 0.0)
        cap = np.clip(cap, 0, None)
        
        c = np.array(obs["graph_now.c"], dtype=float)
        tariff = np.array(obs["graph_now.tariff"], dtype=float)
        
        stock = np.array(obs["stock.qty"], dtype=float)
        stock = np.clip(stock, 0, None)
        
        backlog = np.array(obs["backlog.qty"], dtype=float)
        backlog = np.clip(backlog, 0, None)
        
        demand_forecast = np.array(obs["demand_forecast.qty"], dtype=float)
        demand_forecast = np.clip(demand_forecast, 0, None)
        
        mask = np.array(obs["action_mask"], dtype=float)
        
        n = len(self.slot_edge)
        flows = np.zeros(n)
        
        # Compute capacity per slot
        slot_cap = np.zeros(n)
        for i in range(n):
            lane = self.slot_lane[i]
            if lane is None or lane >= len(self.lane_edges):
                c_slot = cap[self.slot_edge[i]] if self.slot_edge[i] < len(cap) else 0.0
            else:
                lane_edges = self.lane_edges[lane]
                c_slot = min((cap[e] for e in lane_edges if e < len(cap)), default=0.0) if lane_edges else 0.0
            slot_cap[i] = max(c_slot, 0.0)
        
        # Compute demand pressure per slot
        demand_pressure = np.zeros(n)
        for i in range(n):
            edge_head = self.edge_head[self.slot_edge[i]] if self.slot_edge[i] < len(self.edge_head) else -1
            k = int(self.slot_k[i])
            key = (int(edge_head), int(k))
            if key in self.demand_idx:
                d_idx = self.demand_idx[key]
                # Strong weight on backlog, moderate on near-term forecast
                press = backlog[d_idx] * 2.0
                for h in range(min(3, demand_forecast.shape[1])):
                    press += demand_forecast[d_idx, h] * (0.7 ** h)
                demand_pressure[i] = press
        
        # Compute cost per slot
        slot_cost = np.zeros(n)
        for i in range(n):
            edge_idx = self.slot_edge[i]
            k = int(self.slot_k[i])
            if edge_idx < len(c) and k < tariff.shape[1]:
                cost_val = c[edge_idx] + tariff[edge_idx, k]
                slot_cost[i] = max(cost_val, 0.0)
        
        # Greedy allocation: process all slots by demand_pressure / (1 + cost)
        slot_priority = np.zeros(n)
        for i in range(n):
            slot_priority[i] = demand_pressure[i] / (1.0 + slot_cost[i])
        
        # Sort all slots by priority
        sorted_slots = np.argsort(-slot_priority)
        
        # Track available stock per (tail, k)
        available_stock = {}
        for key, idx in self.stock_idx.items():
            available_stock[key] = max(stock[idx], 0.0)
        
        # Allocate greedily
        for slot_idx in sorted_slots:
            if slot_cap[slot_idx] <= 0:
                continue
            
            tail_node = self.edge_tail[self.slot_edge[slot_idx]] if self.slot_edge[slot_idx] < len(self.edge_tail) else -1
            k = int(self.slot_k[slot_idx])
            key = (int(tail_node), int(k))
            
            if key in available_stock:
                request = min(slot_cap[slot_idx], available_stock[key])
                if request > 0:
                    flows[slot_idx] = request
                    available_stock[key] -= request
        
        flows = flows * mask
        
        return {"flows": flows, "override_qty": np.zeros(self.n_over), "release_mode": np.zeros(self.n_rel, dtype=np.int64)}