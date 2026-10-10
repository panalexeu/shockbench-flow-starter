# 0.4699062375097181
import numpy as np

class Agent:
    """Backlog-threshold with binary boost: simple and effective."""
    
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
        
        backlog = observation["backlog.qty"]
        num_commodities = len(self.commodities["id"])
        
        # Aggregate backlog per commodity
        backlog_per_k = np.zeros(num_commodities)
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(backlog):
                k_idx = self.sinks["k"][sink_idx]
                backlog_per_k[k_idx] += backlog[sink_idx]
        
        # Binary logic: if any backlog > 0, boost that commodity's routes to 1.15x
        boost_per_k = np.where(backlog_per_k > 0, 1.15, 1.0)
        
        for slot_idx in range(len(flows)):
            k_idx = self.action_slots["k"][slot_idx]
            flows[slot_idx] *= boost_per_k[k_idx]
        
        flows = flows * action_mask
        
        # Soft cap
        allowed_total = np.sum(self.capacity * action_mask)
        flows_total = np.sum(flows)
        if flows_total > allowed_total * 1.2:
            flows = flows * (allowed_total * 1.15 / (flows_total + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }