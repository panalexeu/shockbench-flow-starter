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
        self.edge_tau = list(edges["tau0"])
        
        self.stock_idx = {}
        for r, (nd, k) in enumerate(config["layout"]["stock_slots"]):
            self.stock_idx[(int(nd), int(k))] = r
        
        self.demand_idx = {}
        for r, (nd, k) in enumerate(config["layout"]["demands"]):
            self.demand_idx[(int(nd), int(k))] = r
        
        self.sinks_pi = np.array([float(x) for x in st["sinks"]["pi"]], dtype=float)

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
        
        # Pipeline visibility: shipments in transit
        pipeline_qty = np.array(obs["pipeline.qty"], dtype=float)
        pipeline_arrival = np.array(obs["pipeline.arrival_week"], dtype=int) if "pipeline.arrival_week" in obs else np.array([])
        pipeline_edge = np.array(obs["pipeline.edge"], dtype=int) if "pipeline.edge" in obs else np.array([])
        pipeline_observed = np.array(obs["pipeline.qty.observed"]) if "pipeline.qty.observed" in obs else np.ones_like(pipeline_qty)
        
        # WIP: work in process that will become stock
        wip_qty = np.array(obs["wip.qty"], dtype=float)
        wip_out_week = np.array(obs["wip.out_week"], dtype=int) if "wip.out_week" in obs else np.array([])
        wip_k = np.array(obs["wip.k"], dtype=int) if "wip.k" in obs else np.array([])
        
        mask = np.array(obs["action_mask"], dtype=float)
        
        n = len(self.slot_edge)
        flows = np.zeros(n)
        
        # Compute slot capacities
        slot_cap = np.zeros(n)
        for i in range(n):
            lane = self.slot_lane[i]
            if lane is None or lane >= len(self.lane_edges):
                c_slot = cap[self.slot_edge[i]] if self.slot_edge[i] < len(cap) else 0.0
            else:
                lane_edges = self.lane_edges[lane]
                if lane_edges:
                    c_slot = min(cap[e] for e in lane_edges if e < len(cap))
                else:
                    c_slot = 0.0
            slot_cap[i] = max(c_slot, 0.0)
        
        # Compute cost per slot
        slot_cost = np.zeros(n)
        for i in range(n):
            edge_idx = self.slot_edge[i]
            k = int(self.slot_k[i])
            if edge_idx < len(c) and k < tariff.shape[1]:
                cost_val = c[edge_idx] + tariff[edge_idx, k]
                slot_cost[i] = max(cost_val, 0.0)
        
        # Enhanced demand pressure: backlog + forecast + pipeline visibility
        demand_pressure = np.zeros(n)
        for i in range(n):
            edge_head = self.edge_head[self.slot_edge[i]] if self.slot_edge[i] < len(self.edge_head) else -1
            k = int(self.slot_k[i])
            key = (int(edge_head), int(k))
            if key in self.demand_idx:
                d_idx = self.demand_idx[key]
                # Backlog is most critical (penalty multiplier drives this)
                press = backlog[d_idx] * 3.0
                # Forecast: immediate > near-term > medium-term
                for h in range(min(3, demand_forecast.shape[1])):
                    press += demand_forecast[d_idx, h] * (0.8 ** h)
                demand_pressure[i] = press
        
        # Group slots by source and allocate greedily
        groups = {}
        for i in range(n):
            tail_idx = self.slot_edge[i]
            tail_node = self.edge_tail[tail_idx] if tail_idx < len(self.edge_tail) else -1
            k = int(self.slot_k[i])
            key = (int(tail_node), int(k))
            if key in self.stock_idx:
                groups.setdefault(key, []).append(i)
        
        for key, idxs in groups.items():
            avail = max(stock[self.stock_idx[key]], 0.0)
            
            if avail > 0:
                # Sort by: high demand pressure, then low cost
                idxs_sorted = sorted(idxs, key=lambda i: (-demand_pressure[i], slot_cost[i]))
                
                remaining = avail
                for i in idxs_sorted:
                    slot_request = min(slot_cap[i], remaining)
                    if slot_request > 0:
                        flows[i] = slot_request
                        remaining -= slot_request
        
        flows = flows * mask
        
        return {"flows": flows, "override_qty": np.zeros(self.n_over), "release_mode": np.zeros(self.n_rel, dtype=np.int64)}