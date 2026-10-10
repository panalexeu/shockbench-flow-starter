# -999
# error: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
import numpy as np

class Agent:
    """Conservative backlog + safe demand forecast: proven 0.47 baseline + weak demand signal."""
    
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
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = self.capacity.copy()
        action_mask = observation["action_mask"]
        
        # Compute backlog per commodity (robust base)
        backlog = observation["backlog.qty"]
        num_commodities = len(self.commodities["id"])
        backlog_per_commodity = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(backlog):
                k_idx = self.sinks["k"][sink_idx]
                backlog_per_commodity[k_idx] += backlog[sink_idx]
        
        # Add weak demand signal: current week only
        demand_forecast = observation["demand_forecast.qty"]
        demand_obs = observation["demand_forecast.qty.observed"]
        demand_per_commodity = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(demand_obs) and demand_obs[sink_idx]:
                k_idx = self.sinks["k"][sink_idx]
                if sink_idx < demand_forecast.shape[0]:
                    demand_per_commodity[k_idx] += demand_forecast[sink_idx, 0]
        
        # Combined urgency: backlog dominates
        urgency = backlog_per_commodity * 2.0 + demand_per_commodity
        max_urgency = np.max(urgency)
        
        if max_urgency > 1e-6:
            urgency_normalized = urgency / (max_urgency + 1e-9)
        else:
            urgency_normalized = np.ones(num_commodities) / num_commodities
        
        # Apply boost: 1.0 to 1.15x based on urgency
        flows = flows * action_mask
        for slot_idx in range(len(flows)):
            k_idx = self.action_slots["k"][slot_idx]
            boost = 1.0 + 0.15 * urgency_normalized[k_idx]
            flows[slot_idx] *= boost
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }