# 0.4699062375097181
import numpy as np

class Agent:
    """Conservative lead-time plus backlog: front-load long routes only when backlog high."""
    
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
        flows = self.capacity.copy()
        action_mask = observation["action_mask"]
        
        # Apply mask
        flows = flows * action_mask
        
        # Backlog per commodity
        backlog = observation["backlog.qty"]
        num_commodities = len(self.commodities["id"])
        backlog_per_commodity = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(backlog):
                k_idx = self.sinks["k"][sink_idx]
                backlog_per_commodity[k_idx] += backlog[sink_idx]
        
        # Check if backlog is high (trigger lead-time boost)
        total_backlog = np.sum(backlog_per_commodity)
        high_backlog_threshold = 10.0  # Tunable
        enable_lead_time_boost = (total_backlog > high_backlog_threshold)
        
        # Lead times
        edge_tau = np.array(self.edges["tau0"], dtype=float)
        
        # Apply boosts
        for slot_idx in range(len(flows)):
            k_idx = self.action_slots["k"][slot_idx]
            edge_idx = self.action_slots["edge"][slot_idx]
            
            # Backlog boost (always on)
            max_backlog = np.max(backlog_per_commodity)
            if max_backlog > 1e-6:
                backlog_boost = 1.0 + 0.10 * (backlog_per_commodity[k_idx] / (max_backlog + 1e-9))
            else:
                backlog_boost = 1.0
            
            # Lead-time boost (conditional on high backlog)
            lead_time_boost = 1.0
            if enable_lead_time_boost and edge_idx < len(edge_tau):
                tau = edge_tau[edge_idx]
                if tau > 2.0:  # Only boost on routes with lead time > 2 weeks
                    lead_time_boost = 1.0 + 0.05 * min(tau / 6.0, 1.0)  # Up to 1.05x
            
            flows[slot_idx] *= (backlog_boost * lead_time_boost)
        
        # Soft cap to avoid exceeding reasonable bounds
        mask_capacity = self.capacity * action_mask
        total_flows = np.sum(flows)
        max_allowed = np.sum(mask_capacity) * 1.25
        if total_flows > max_allowed:
            flows = flows * (max_allowed / (total_flows + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }