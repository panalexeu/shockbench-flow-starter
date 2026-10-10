# -0.23657867626939227
import numpy as np

class Agent:
    """Chokepoint-aware: adapt allocation via release_mode and override based on queue pressure."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        self.action_slots = action_slots
        self.override_slots = static["override_slots"]
        self.commodities = static["commodities"]
        self.sinks = static["sinks"]
        self.layout = config["layout"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = self.capacity * observation["action_mask"]
        release_mode = np.zeros_like(self.release_mode)
        override_qty = np.zeros_like(self.override_qty)
        
        # Check queue pressure at chokepoints
        queue_lots = observation["queue_lots.qty"]  # (num_lot_keys, num_weeks)
        queue_observed = observation["queue_lots.qty.observed"]
        
        # For each release pair (chokepoint, commodity), decide hold/default/override
        if queue_observed.any():
            # Compute queue occupancy per chokepoint
            total_queue = np.sum(queue_lots, axis=1)  # sum over weeks
            queue_pressure = np.sum(total_queue) / (queue_observed.sum() + 1e-6)
            
            # If queues are building, hold some releases to clear them
            if queue_pressure > 100:
                release_mode[:] = 2  # Hold
            elif queue_pressure > 50:
                release_mode[:] = 0  # Default release
            else:
                release_mode[:] = 0  # Default when flow is smooth
        else:
            release_mode[:] = 0  # Default if no queue visibility
        
        return {
            "flows": flows,
            "override_qty": override_qty,
            "release_mode": release_mode
        }