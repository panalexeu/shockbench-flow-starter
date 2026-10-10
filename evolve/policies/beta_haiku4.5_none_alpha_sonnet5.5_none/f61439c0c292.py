# 0.5475567601348417
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
        self.tau0 = np.array([int(x) if x is not None else 1 for x in edges["tau0"]], dtype=int)
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
        
        c = np.array(obs["graph_now.c"], dtype=float)
        tariff = np.array(obs["graph_now.tariff"], dtype=float)
        tau = np.array(obs["graph_now.tau"], dtype=int) if "graph_now.tau" in obs else self.tau0
        war_risk = np.array(obs["graph_now.war_risk"], dtype=int) if "graph_now.war_risk" in obs else np.zeros(len(self.chokepoints), dtype=int)
        
        stock = np.array(obs["stock.qty"], dtype=float)
        stock = np.clip(stock, 0, None)
        
        backlog = np.array(obs["backlog.qty"], dtype=float)
        backlog = np.clip(backlog, 0, None)
        
        demand_forecast = np.array(obs["demand_forecast.qty"], dtype=float)
        demand_forecast = np.clip(demand_forecast, 0, None)
        
        # Pipeline in-flight stock
        pipeline_in_flight = {}
        if "pipeline.qty" in obs and "pipeline.edge" in obs and "pipeline.k" in obs:
            pipeline_qty = np.array(obs["pipeline.qty"], dtype=float)
            pipeline_edge = np.array(obs["pipeline.edge"], dtype=int)
            pipeline_k = np.array(obs["pipeline.k"], dtype=int)
            pipeline_observed = np.array(obs["pipeline.qty.observed"]) if "pipeline.qty.observed" in obs else np.ones_like(pipeline_qty)
            for idx in range(len(pipeline_qty)):
                if pipeline_observed[idx] > 0:
                    edge_idx = pipeline_edge[idx]
                    if edge_idx < len(self.edge_head):
                        head_node = self.edge_head[edge_idx]
                        k_val = pipeline_k[idx]
                        key = (int(head_node), int(k_val))
                        pipeline_in_flight[key] = pipeline_in_flight.get(key, 0.0) + pipeline_qty[idx]
        
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
        
        # Cost per slot
        slot_cost = np.zeros(n)
        for i in range(n):
            edge_idx = self.slot_edge[i]
            k = self.slot_k[i]
            if edge_idx < len(c) and k < tariff.shape[1]:
                cost_val = c[edge_idx] + tariff[edge_idx, k]
                slot_cost[i] = max(cost_val, 0.0)
        
        # Demand with lead-time-aware decay
        demand_pressure = np.zeros(n)
        for i in range(n):
            edge_head = self.edge_head[self.slot_edge[i]] if self.slot_edge[i] < len(self.edge_head) else -1
            k = self.slot_k[i]
            edge_idx = self.slot_edge[i]
            tau_val = int(tau[edge_idx]) if edge_idx < len(tau) else 1
            
            key = (int(edge_head), int(k))
            if key in self.demand_idx:
                d_idx = self.demand_idx[key]
                # Strong weight on backlog, decaying weight on forecast
                press = backlog[d_idx] * 2.5
                # Use lead time to window the relevant forecast horizon
                for h in range(min(tau_val + 1, demand_forecast.shape[1])):
                    decay = 1.0 / (1.0 + 0.3 * h)
                    press += demand_forecast[d_idx, h] * decay
                demand_pressure[i] = press
        
        # Reduce pressure if in-flight stock is substantial
        for i in range(n):
            edge_head = self.edge_head[self.slot_edge[i]] if self.slot_edge[i] < len(self.edge_head) else -1
            k = self.slot_k[i]
            key = (int(edge_head), int(k))
            in_flight = pipeline_in_flight.get(key, 0.0)
            if in_flight > 0:
                demand_pressure[i] = max(demand_pressure[i] - in_flight * 0.6, 0.0)
        
        # Group by source
        groups = {}
        for i in range(n):
            tail_node = self.edge_tail[self.slot_edge[i]] if self.slot_edge[i] < len(self.edge_tail) else -1
            k = self.slot_k[i]
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
                flows[i] = slot_request
                remaining -= slot_request
        
        flows = flows * mask
        
        # Tanker release
        override_qty = np.zeros(self.n_over)
        release_mode = np.zeros(self.n_rel, dtype=np.int64)
        
        for j in range(self.n_rel):
            if j < len(self.chokepoints):
                chp = self.chokepoints[j]
                if chp < len(war_risk) and war_risk[chp] > 0:
                    release_mode[j] = 2
                else:
                    release_mode[j] = 0
        
        return {"flows": flows, "override_qty": override_qty, "release_mode": release_mode}