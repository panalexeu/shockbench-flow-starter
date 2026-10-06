import numpy as np

class Agent:
    def __init__(self, config=None):
        static = config["static"]  
        action = config["spaces"]["action"]
        u0 = static["edges"]["u0"] 
        self.capacity = np.array([u0[e] for e in static["action_slots"]["edge"]], dtype=float)
        self.override_qty = np.zeros(action["override_qty"]["shape"])
        self.release_mode = np.zeros(action["release_mode"]["shape"], dtype=np.int64)  
        self.rng = np.random.default_rng(config["policy_seed"])

    def act(self, observation):
        flows = self.capacity * observation["action_mask"]
        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}