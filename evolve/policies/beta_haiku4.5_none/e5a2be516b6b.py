# 0.4699062375097181
import numpy as np

class Agent:
    """Hybrid: base capacity + backlog-driven boost on high-urgency routes."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        self.action_slots = action_slots
        self.edges = static["edges"]
        self.sinks = static["sinks"]
        self.commodities = static["commodities"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = self.capacity.copy()  # Start with full capacity
        action_mask = observation["action_mask"]
        
        # Compute commodity-level backlog pressure
        backlog = observation["backlog.qty"]
        num_commodities = len(self.commodities["id"])
        backlog_per_commodity = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(backlog):
                k_idx = self.sinks["k"][sink_idx]
                backlog_per_commodity[k_idx] += backlog[sink_idx]
        
        # Compute average backlog
        avg_backlog = np.mean(backlog_per_commodity[backlog_per_commodity > 0]) if np.any(backlog_per_commodity > 0) else 0
        
        # Apply mask and mild backlog boost
        flows = flows * action_mask
        
        # Boost flow on high-backlog commodities
        for slot_idx in range(len(flows)):
            k_idx = self.action_slots["k"][slot_idx]
            if backlog_per_commodity[k_idx] > avg_backlog * 1.5:
                flows[slot_idx] *= 1.05  # Mild boost
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }