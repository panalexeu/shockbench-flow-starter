# 0.4699062375097181
import numpy as np

class Agent:
    """Refined baseline: send capacity with demand-aware scaling."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        layout = config["layout"]
        
        # Extract nominal capacities for each action slot
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        # Initialize override and release controls
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
        
        self.action_slots = action_slots
        self.sinks = static["sinks"]
        self.commodities = static["commodities"]
    
    def act(self, observation):
        # Base strategy: send full capacity where allowed
        flows = self.capacity * observation["action_mask"]
        
        # Light demand adjustment: if current demand is very high, scale up slightly
        demand_forecast = observation["demand_forecast.qty"]  # (num_demands, 8)
        if observation["demand_forecast.qty.observed"].any():
            current_demand = np.sum(demand_forecast[:, 0])
            backlog = np.sum(observation["backlog.qty"])
            if backlog > current_demand * 0.1:  # Significant backlog: send more aggressively
                flows = flows * 1.1
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }