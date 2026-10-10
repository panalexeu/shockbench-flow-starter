# 0.4699062375097181
import numpy as np

class Agent:
    """Baseline: send full capacity on allowed slots."""
    
    def __init__(self, config=None):
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        # Extract nominal capacities for each action slot
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        # Initialize override and release controls
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        # Send up to capacity where not prohibited
        flows = self.capacity * observation["action_mask"]
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }