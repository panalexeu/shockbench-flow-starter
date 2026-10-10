# 0.4699062375097181
import numpy as np

class Agent:
    """Optimized baseline: capacity + responsive backlog boost."""
    
    def __init__(self, config=None):
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
        flows = flows * action_mask
        
        # Compute backlog per commodity
        backlog = observation["backlog.qty"]
        num_commodities = len(self.commodities["id"])
        backlog_per_k = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(backlog):
                k_idx = self.sinks["k"][sink_idx]
                backlog_per_k[k_idx] += backlog[sink_idx]
        
        # Responsive backlog boost
        if np.max(backlog_per_k) > 0:
            max_backlog = np.max(backlog_per_k)
            backlog_pressure = backlog_per_k / (max_backlog + 1e-9)
            
            for slot_idx in range(len(flows)):
                k_idx = self.action_slots["k"][slot_idx]
                # Boost high-backlog commodities: 1.0 to 1.20x
                boost = 1.0 + 0.20 * backlog_pressure[k_idx]
                flows[slot_idx] *= boost
        
        # Soft capacity cap
        allowed_capacity = self.capacity * action_mask
        total_allowed = np.sum(allowed_capacity)
        total_flows = np.sum(flows)
        if total_flows > total_allowed * 1.3 and total_allowed > 0:
            flows = flows * (total_allowed * 1.25 / (total_flows + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }