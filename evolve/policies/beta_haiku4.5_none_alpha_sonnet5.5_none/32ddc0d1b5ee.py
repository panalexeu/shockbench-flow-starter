# 0.5379137367332142
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
        self.commodities_v = np.array([float(x) for x in st["commodities"]["v"]], dtype=float)
        
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
        
        # Account for pipeline: shipments in transit reduce effective demand
        pipeline_by_sink = {}
        if "pipeline.qty" in obs and "pipeline.arrival_week" in obs:
            pipeline_qty = np.array(obs["pipeline.qty"], dtype=float)
            pipeline_edge = np.array(obs["pipeline.edge"], dtype=int)
            pipeline_k = np.array(obs["pipeline.k"], dtype=int)
            pipeline_arrival = np.array(obs["pipeline.arrival_week"], dtype=int)
            pipeline_observed = np.array(obs["pipeline.qty.observed"]) if "pipeline.qty.observed" in obs else np.ones_like(pipeline_qty)
            
            current_week = int(obs["week"][0]) if "week" in obs else 1
            for idx in range(len(pipeline_qty)):
                if pipeline_observed[idx] > 0 and pipeline_arrival[idx] <= current_week + 1:
                    edge_idx = pipeline_edge[idx]
                    if edge_idx < len(self.edge_head):
                        sink_node = self.edge_head[edge_idx]
                        k = pipeline_k[idx]
                        key = (int(sink_node), int(k))
                        pipeline_by_sink[key] = pipeline_by_sink.get(key, 0.0) + pipeline_qty[idx]
        
        mask = np.array(obs["action_mask"], dtype=float)
        
        n = len(self.slot_edge)
        flows = np.zeros(n)
        
        # Slot capacity
        slot_cap = np.zeros(n)
        for i in range(n):
            lane = self.slot_lane[i]
            if lane is None or lane >= len(self.lane_edges):
                c_slot = cap[self.slot_edge[i]] if self.slot_edge[i] < len(cap) else 0.0
            else:
                lane_edges = self.lane_edges[lane]
                c_slot = min((cap[e] for e in lane_edges if e < len(cap)), default=0.0) if lane_edges else 0.0
            slot_cap[i] = max(c_slot, 0.0)
        
        # Demand pressure: backlog + forecast, adjusted by pipeline arrivals
        demand_pressure = np.zeros(n)
        for i in range(n):
            edge_head = self.edge_head[self.slot_edge[i]] if self.slot_edge[i] < len(self.edge_head) else -1
            k = int(self.slot_k[i])
            key = (int(edge_head), int(k))
            if key in self.demand_idx:
                d_idx = self.demand_idx[key]
                press = backlog[d_idx] * 2.5
                # Reduce demand pressure by expected pipeline arrivals
                pipeline_inflow = pipeline_by_sink.get(key, 0.0)
                for h in range(min(4, demand_forecast.shape[1])):
                    press += max(0, demand_forecast[d_idx, h] - pipeline_inflow / 4.0) * np.exp(-0.2 * h)
                demand_pressure[i] = max(press, 0.0)
        
        # Cost per slot
        slot_cost = np.zeros(n)
        for i in range(n):
            edge_idx = self.slot_edge[i]
            k = int(self.slot_k[i])
            if edge_idx < len(c) and k < tariff.shape[1]:
                cost_val = c[edge_idx] + tariff[edge_idx, k]
                slot_cost[i] = max(cost_val, 0.0)
        
        # Group by source
        groups = {}
        for i in range(n):
            tail_node = self.edge_tail[self.slot_edge[i]] if self.slot_edge[i] < len(self.edge_tail) else -1
            k = int(self.slot_k[i])
            key = (int(tail_node), int(k))
            if key in self.stock_idx:
                groups.setdefault(key, []).append(i)
        
        # Allocate
        for key, idxs in groups.items():
            avail = max(stock[self.stock_idx[key]], 0.0)
            idxs_sorted = sorted(idxs, key=lambda i: (-demand_pressure[i], slot_cost[i]))
            
            remaining = avail
            for i in idxs_sorted:
                slot_request = min(slot_cap[i], remaining)
                if slot_request > 0:
                    flows[i] = slot_request
                    remaining -= slot_request
        
        flows = flows * mask
        
        return {"flows": flows, "override_qty": np.zeros(self.n_over), "release_mode": np.zeros(self.n_rel, dtype=np.int64)}