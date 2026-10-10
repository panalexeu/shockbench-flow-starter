# 0.2492089785473344
import numpy as np

class Agent:
    """Demand-aware routing with backlog reduction focus."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        layout = config["layout"]
        
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        self.action_slots = action_slots
        self.edges = static["edges"]
        self.sinks = static["sinks"]
        self.commodities = static["commodities"]
        self.layout_demands = layout["demands"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        action_mask = observation["action_mask"]
        flows = self.capacity.copy() * action_mask
        
        # Compute backlog pressure per commodity
        backlog = observation["backlog.qty"]
        backlog_obs = observation["backlog.qty.observed"]
        num_commodities = len(self.commodities["id"])
        backlog_per_k = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(backlog) and backlog_obs[sink_idx]:
                k_idx = self.sinks["k"][sink_idx]
                backlog_per_k[k_idx] += backlog[sink_idx]
        
        max_backlog = np.max(backlog_per_k) if np.max(backlog_per_k) > 0 else 1.0
        backlog_norm = backlog_per_k / (max_backlog + 1e-9)
        
        # Compute near-term demand (weeks 0-3)
        demand_forecast = observation["demand_forecast.qty"]
        demand_forecast_obs = observation["demand_forecast.qty.observed"]
        demand_per_k = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.layout_demands)):
            if sink_idx < len(demand_forecast) and demand_forecast_obs[sink_idx].any():
                k_idx = self.layout_demands[sink_idx][1]
                demand_per_k[k_idx] += np.sum(demand_forecast[sink_idx, :4])
        
        max_demand = np.max(demand_per_k) if np.max(demand_per_k) > 0 else 1.0
        demand_norm = demand_per_k / (max_demand + 1e-9)
        
        # Combined urgency: prioritize reducing backlog
        urgency = 0.7 * backlog_norm + 0.3 * demand_norm
        
        # Boost per slot based on commodity urgency
        for slot_idx in range(len(flows)):
            if action_mask[slot_idx]:
                k_idx = self.action_slots["k"][slot_idx]
                # High urgency = send more
                flows[slot_idx] *= (0.8 + 0.4 * urgency[k_idx])
        
        # Soft cap to avoid over-shooting
        allowed_total = np.sum(self.capacity * action_mask)
        flows_total = np.sum(flows)
        if flows_total > allowed_total * 1.15:
            flows = flows * (allowed_total * 1.10 / (flows_total + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }