# 0.4699062375097181
import numpy as np

class Agent:
    """Threat-aware with fixed boosting logic (no ambiguous array comparisons)."""
    
    def __init__(self, config=None):
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        self.action_slots = action_slots
        self.commodities = static["commodities"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = self.capacity.copy()
        action_mask = observation["action_mask"]
        week = int(observation["week"][0])
        
        # Detect pending prohibitions and build threat map
        pending_edge = observation["pending_prohibitions.edge"]
        pending_k = observation["pending_prohibitions.k"]
        pending_week = observation["pending_prohibitions.effective_week"]
        pending_obs = observation["pending_prohibitions.edge.observed"]
        
        # Build a dict of (edge, k) -> boost_factor
        threat_map = {}
        for i in range(len(pending_obs)):
            if pending_obs[i] == 1:  # Safely check observed flag
                edge_idx = int(pending_edge[i])
                k_idx = int(pending_k[i])
                eff_week = int(pending_week[i])
                weeks_until = max(0, eff_week - week)
                
                if 0 < weeks_until <= 1:
                    boost = 1.5
                elif 1 < weeks_until <= 2:
                    boost = 1.3
                elif 2 < weeks_until <= 4:
                    boost = 1.15
                else:
                    boost = 1.0
                
                key = (edge_idx, k_idx)
                threat_map[key] = max(threat_map.get(key, 1.0), boost)
        
        # Apply threat boost
        for slot_idx in range(len(flows)):
            edge_idx = self.action_slots["edge"][slot_idx]
            k_idx = self.action_slots["k"][slot_idx]
            key = (edge_idx, k_idx)
            boost = threat_map.get(key, 1.0)
            flows[slot_idx] *= boost
        
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