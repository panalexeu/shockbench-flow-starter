# 0.37986577423938317
import numpy as np

class Agent:
    """Demand-forecast driven: weight near-term forecast into allocation."""
    
    def __init__(self, config=None):
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        self.action_slots = action_slots
        self.sinks = static["sinks"]
        self.commodities = static["commodities"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = self.capacity.copy()
        action_mask = observation["action_mask"]
        flows = flows * action_mask
        
        # Combine backlog and near-term forecast into demand signal
        backlog = observation["backlog.qty"]
        demand_forecast = observation["demand_forecast.qty"]
        
        num_commodities = len(self.commodities["id"])
        demand_signal = np.zeros(num_commodities)
        
        # Weight weeks 0-2 more heavily
        forecast_weights = np.array([1.5, 1.2, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1])
        
        for sink_idx in range(len(self.sinks["k"])):
            k_idx = self.sinks["k"][sink_idx]
            if sink_idx < len(demand_forecast):
                demand_signal[k_idx] += np.sum(demand_forecast[sink_idx, :] * forecast_weights)
            if sink_idx < len(backlog):
                demand_signal[k_idx] += backlog[sink_idx] * 2.0
        
        # Normalize demand signal
        total_demand = np.sum(demand_signal)
        if total_demand > 0:
            demand_norm = demand_signal / total_demand
        else:
            demand_norm = np.ones(num_commodities) / num_commodities
        
        # Allocate based on demand share: 0.90 to 1.15x
        for slot_idx in range(len(flows)):
            k_idx = self.action_slots["k"][slot_idx]
            demand_factor = 0.90 + 0.25 * min(demand_norm[k_idx] * len(self.commodities["id"]), 1.0)
            flows[slot_idx] *= demand_factor
        
        # Soft capacity cap
        allowed_capacity = self.capacity * action_mask
        total_allowed = np.sum(allowed_capacity)
        total_flows = np.sum(flows)
        if total_flows > total_allowed * 1.3 and total_allowed > 0:
            flows = flows * (total_allowed * 1.22 / (total_flows + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }