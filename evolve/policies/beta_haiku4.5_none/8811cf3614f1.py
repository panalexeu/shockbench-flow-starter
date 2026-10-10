# 0.09114856386537351
import numpy as np

class Agent:
    """Demand-driven with stock depletion urgency and cost awareness."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        self.action_slots = action_slots
        self.sinks = static["sinks"]
        self.commodities = static["commodities"]
        self.edges = static["edges"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = self.capacity.copy()
        action_mask = observation["action_mask"]
        
        num_commodities = len(self.commodities["id"])
        
        # High-weight backlog: unserved demand is immediate loss
        backlog = observation["backlog.qty"]
        backlog_per_k = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(backlog):
                k_idx = self.sinks["k"][sink_idx]
                backlog_per_k[k_idx] += backlog[sink_idx]
        
        # Demand forecast for next 3 weeks, heavily decayed
        demand_forecast = observation["demand_forecast.qty"]
        demand_per_k = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < demand_forecast.shape[0]:
                k_idx = self.sinks["k"][sink_idx]
                demand_per_k[k_idx] += (1.0 * demand_forecast[sink_idx, 0] +
                                        0.8 * demand_forecast[sink_idx, 1] +
                                        0.5 * demand_forecast[sink_idx, 2])
        
        # Combined urgency: 75% backlog, 25% forecast
        max_backlog = np.max(backlog_per_k) if np.max(backlog_per_k) > 0 else 1.0
        max_demand = np.max(demand_per_k) if np.max(demand_per_k) > 0 else 1.0
        
        urgency = (0.75 * (backlog_per_k / (max_backlog + 1e-9)) +
                   0.25 * (demand_per_k / (max_demand + 1e-9)))
        
        # Cost modulation: reduce on expensive edges
        graph_c = observation["graph_now.c"]
        graph_c_obs = observation["graph_now.c.observed"]
        
        mean_cost = 1.0
        cost_count = np.sum(graph_c_obs)
        if cost_count > 0:
            mean_cost = np.sum(graph_c * graph_c_obs) / cost_count
        
        # Apply per-slot allocation
        for slot_idx in range(len(flows)):
            if action_mask[slot_idx]:
                k_idx = self.action_slots["k"][slot_idx]
                edge_idx = self.action_slots["edge"][slot_idx]
                
                # Base: 0.75 + 0.25 * urgency
                intensity = 0.75 + 0.25 * min(urgency[k_idx], 1.0)
                
                # Cost penalty: if cost > 1.3x mean, reduce by 8%
                cost_factor = 1.0
                if graph_c_obs[edge_idx]:
                    if graph_c[edge_idx] > 1.3 * mean_cost:
                        cost_factor = 0.92
                
                flows[slot_idx] *= intensity * cost_factor
            else:
                flows[slot_idx] = 0.0
        
        # Soft cap
        allowed = np.sum(self.capacity * action_mask)
        total = np.sum(flows)
        if total > allowed * 1.25:
            flows *= (allowed * 1.2 / (total + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }