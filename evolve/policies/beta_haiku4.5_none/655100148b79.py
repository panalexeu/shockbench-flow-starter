# 0.31456237528312153
import numpy as np

class Agent:
    """Demand-forecast priority: allocate more to high-demand commodities while respecting capacity."""
    
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
        flows = np.zeros_like(self.capacity)
        action_mask = observation["action_mask"]
        
        # Extract signals
        demand_forecast = observation["demand_forecast.qty"]  # (num_demands, 8)
        backlog = observation["backlog.qty"]
        
        # Compute commodity urgency from next week demand + backlog
        num_commodities = len(self.commodities["id"])
        urgency = np.zeros(num_commodities)
        
        # Sum forecast at h=1 (next week) with some current week weight
        for sink_idx in range(len(self.sinks["k"])):
            k_idx = self.sinks["k"][sink_idx]
            if sink_idx < len(demand_forecast):
                # Weight next week heavily, current week lighter
                urgency[k_idx] += demand_forecast[sink_idx, 0] * 0.5 + demand_forecast[sink_idx, 1] * 1.0
        
        # Add backlog pressure
        for sink_idx in range(len(self.sinks["k"])):
            k_idx = self.sinks["k"][sink_idx]
            if sink_idx < len(backlog):
                urgency[k_idx] += backlog[sink_idx] * 2.0
        
        # Normalize urgency
        total_urgency = np.sum(urgency)
        if total_urgency > 0:
            urgency_norm = urgency / total_urgency
        else:
            urgency_norm = np.ones(num_commodities) / num_commodities
        
        # Allocate with modest priority weighting (0.85 to 1.15 range)
        for slot_idx in range(len(flows)):
            if action_mask[slot_idx]:
                k_idx = self.action_slots["k"][slot_idx]
                # Base 0.85, boost by urgency (up to 0.30 extra)
                factor = 0.85 + 0.30 * min(urgency_norm[k_idx] * len(self.commodities["id"]), 1.0)
                flows[slot_idx] = self.capacity[slot_idx] * factor
        
        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}