# 0.4699062375097186
import numpy as np

class Agent:
    """Maximalist: Send higher allocation on allowed slots with minimal modulation."""
    
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
        
        # Base: send up to 1.25x capacity on allowed slots, capped by actual allowed capacity
        flows = flows * action_mask * 1.25
        
        # Hard cap: total flows should not exceed sum of allowed capacities by too much
        allowed_total = np.sum(self.capacity * action_mask)
        flows_total = np.sum(flows)
        if flows_total > allowed_total * 1.25:
            flows = flows * (allowed_total * 1.2 / (flows_total + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }