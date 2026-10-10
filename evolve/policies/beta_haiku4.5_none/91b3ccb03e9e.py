# 0.4699062375097181
import numpy as np

class Agent:
    """Lead-time aware: pre-buffer long routes to compensate for delays."""
    
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
        self.tau0 = np.array(self.edges["tau0"], dtype=float)
        
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
        
        max_backlog = np.max(backlog_per_k) if np.max(backlog_per_k) > 0 else 1.0
        backlog_pressure = backlog_per_k / (max_backlog + 1e-9)
        
        # Lead-time aware allocation
        for slot_idx in range(len(flows)):
            k_idx = self.action_slots["k"][slot_idx]
            edge_idx = self.action_slots["edge"][slot_idx]
            
            # Backlog boost: base 1.0 to 1.18x
            backlog_boost = 1.0 + 0.18 * backlog_pressure[k_idx]
            
            # Lead-time boost: longer routes get extra priority
            lead_time = self.tau0[edge_idx]
            lead_boost = 1.0 + 0.08 * min(lead_time / 6.0, 1.0)
            
            flows[slot_idx] *= (backlog_boost * lead_boost)
        
        # Soft capacity normalization
        allowed_capacity = self.capacity * action_mask
        total_allowed = np.sum(allowed_capacity)
        total_flows = np.sum(flows)
        if total_flows > total_allowed * 1.35 and total_allowed > 0:
            flows = flows * (total_allowed * 1.25 / (total_flows + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }