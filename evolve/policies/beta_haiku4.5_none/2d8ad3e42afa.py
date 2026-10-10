# 0.4699062375097181
import numpy as np

class Agent:
    """Disruption-aware: boost flow on routes threatened by upcoming prohibitions."""
    
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
        week = int(observation["week"][0])
        
        # Detect upcoming prohibitions within 2 weeks
        pending_prohibitions_edge = observation["pending_prohibitions.edge"]
        pending_prohibitions_k = observation["pending_prohibitions.k"]
        pending_prohibitions_week = observation["pending_prohibitions.effective_week"]
        pending_obs = observation["pending_prohibitions.edge.observed"]
        
        # Compute threat score for each edge-commodity pair
        threat_boost = np.ones(len(flows))
        if pending_obs.any():
            for i in np.where(pending_obs)[0]:
                edge_idx = int(pending_prohibitions_edge[i])
                k_idx = int(pending_prohibitions_k[i])
                eff_week = int(pending_prohibitions_week[i])
                weeks_until = max(0, eff_week - week)
                
                # High boost if closure is imminent (1-2 weeks)
                if 0 < weeks_until <= 2:
                    boost_factor = 1.4
                elif 2 < weeks_until <= 4:
                    boost_factor = 1.2
                else:
                    boost_factor = 1.0
                
                # Apply to all slots using this edge and commodity
                for slot_idx in range(len(flows)):
                    if (self.action_slots["edge"][slot_idx] == edge_idx and
                        self.action_slots["k"][slot_idx] == k_idx):
                        threat_boost[slot_idx] = max(threat_boost[slot_idx], boost_factor)
        
        # Apply threat boost and action mask
        flows = flows * threat_boost * action_mask
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }