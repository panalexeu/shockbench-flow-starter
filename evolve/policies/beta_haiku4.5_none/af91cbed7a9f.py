# 0.37986577423938317
import numpy as np

class Agent:
    """Demand-driven allocation with backlog priority."""
    
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
        
        # Extract demand and backlog signals
        demand_forecast = observation["demand_forecast.qty"]  # (num_demands, 8)
        backlog = observation["backlog.qty"]
        
        # Compute commodity urgency: current + next week demand + backlog
        num_commodities = len(self.commodities["id"])
        urgency = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            k_idx = self.sinks["k"][sink_idx]
            if observation["demand_forecast.qty.observed"].any() and sink_idx < len(demand_forecast):
                # Current + next week demand, weighted
                demand_urgency = demand_forecast[sink_idx, 0] + 0.8 * demand_forecast[sink_idx, 1]
                urgency[k_idx] += demand_urgency
            
            if sink_idx < len(backlog):
                # Backlog is very urgent
                urgency[k_idx] += backlog[sink_idx] * 3.0
        
        # Normalize urgency to scale 0-1
        total_urgency = np.sum(urgency) + 1e-6
        urgency_normalized = urgency / total_urgency
        
        # Allocate flows based on urgency, with baseline of 0.9x capacity
        for slot_idx in range(len(flows)):
            if action_mask[slot_idx]:
                k_idx = self.action_slots["k"][slot_idx]
                # Base 0.9, scale up by urgency (up to 1.1)
                allocation_factor = 0.9 + 0.2 * min(urgency_normalized[k_idx] * 5, 1.0)
                flows[slot_idx] = self.capacity[slot_idx] * allocation_factor
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }