# -0.5002840096565679
import numpy as np

class Agent:
    """Lead-time aware with conditional release control."""
    
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
        self.edges = static["edges"]
        self.override_slots = static["override_slots"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = self.capacity.copy()
        action_mask = observation["action_mask"]
        flows = flows * action_mask
        
        # Backlog signal
        backlog = observation["backlog.qty"]
        num_commodities = len(self.commodities["id"])
        backlog_per_k = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(backlog):
                k_idx = self.sinks["k"][sink_idx]
                backlog_per_k[k_idx] += backlog[sink_idx]
        
        total_backlog = np.sum(backlog_per_k)
        
        # Lead time info
        edge_tau = np.array(self.edges["tau0"], dtype=float)
        
        # Apply allocation: base + lead-time boost when backlog exists
        for slot_idx in range(len(flows)):
            k_idx = self.action_slots["k"][slot_idx]
            edge_idx = self.action_slots["edge"][slot_idx]
            
            base_factor = 0.80
            
            # Add backlog urgency boost
            if total_backlog > 1e-6:
                backlog_boost = 0.25 * (backlog_per_k[k_idx] / (np.max(backlog_per_k) + 1e-9))
            else:
                backlog_boost = 0.0
            
            # Lead-time boost: front-load longer routes when backlog is present
            lead_time_boost = 0.0
            if total_backlog > 5.0 and edge_tau[edge_idx] > 2:
                lead_time_boost = 0.08 * min(edge_tau[edge_idx] / 8.0, 1.0)
            
            allocation_factor = base_factor + backlog_boost + lead_time_boost
            flows[slot_idx] *= allocation_factor
        
        # Queue-based release control (if queue data available)
        queue_lots = observation["queue_lots.qty"]
        queue_obs = observation["queue_lots.qty.observed"]
        
        release_mode = np.zeros_like(self.release_mode)
        
        if queue_obs.any():
            total_queue = np.sum(queue_lots[queue_obs.astype(bool)])
            # Default: hold if high queue pressure, otherwise allow default
            if total_queue > 200:
                release_mode[:] = 2  # Hold
            else:
                release_mode[:] = 0  # Default
        else:
            release_mode[:] = 0
        
        # Soft cap
        allowed_capacity = self.capacity * action_mask
        total_allowed = np.sum(allowed_capacity)
        total_flows = np.sum(flows)
        if total_flows > total_allowed * 1.2:
            flows = flows * (total_allowed * 1.15 / (total_flows + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": release_mode
        }