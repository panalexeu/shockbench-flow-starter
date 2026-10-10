# 0.4699062375097181
import numpy as np

class Agent:
    """Refined backlog-driven: full capacity + smart commodity-level boost."""
    
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
        
        # Apply action mask first
        flows = flows * action_mask
        
        # Compute backlog per commodity
        backlog = observation["backlog.qty"]
        num_commodities = len(self.commodities["id"])
        backlog_per_commodity = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(backlog):
                k_idx = self.sinks["k"][sink_idx]
                backlog_per_commodity[k_idx] += backlog[sink_idx]
        
        # Compute pressure: backlog normalized by commodity count
        max_backlog = np.max(backlog_per_commodity)
        if max_backlog > 1e-6:
            backlog_pressure = backlog_per_commodity / (max_backlog + 1e-9)
        else:
            backlog_pressure = np.zeros(num_commodities)
        
        # Apply commodity-level boost: up to 1.15x on high-backlog commodities
        for slot_idx in range(len(flows)):
            k_idx = self.action_slots["k"][slot_idx]
            boost_factor = 1.0 + 0.15 * backlog_pressure[k_idx]
            flows[slot_idx] *= boost_factor
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }