# 0.09639612394858282
import numpy as np

class Agent:
    """Policy 2: Aggressive pre-buffering on routes threatened by imminent prohibitions."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        layout = config["layout"]
        
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        self.action_slots = action_slots
        self.edges = static["edges"]
        self.sinks = static["sinks"]
        self.layout_demands = layout["demands"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = np.zeros_like(self.capacity)
        action_mask = observation["action_mask"]
        week = int(observation["week"][0])
        
        # Detect imminent prohibitions (1-3 weeks out)
        pending_edge = observation["pending_prohibitions.edge"]
        pending_k = observation["pending_prohibitions.k"]
        pending_week = observation["pending_prohibitions.effective_week"]
        pending_obs = observation["pending_prohibitions.edge.observed"]
        
        threatened = {}  # (edge, k) -> boost factor
        if pending_obs.any():
            for i in np.where(pending_obs)[0]:
                edge_idx = int(pending_edge[i])
                k_idx = int(pending_k[i])
                eff_week = int(pending_week[i])
                weeks_until = max(0, eff_week - week)
                
                if 0 < weeks_until <= 1:
                    boost = 1.6
                elif 1 < weeks_until <= 3:
                    boost = 1.4
                else:
                    boost = 1.0
                threatened[(edge_idx, k_idx)] = max(threatened.get((edge_idx, k_idx), 1.0), boost)
        
        # Also boost on currently closing chokepoints
        closure_chokepoint = observation["closure_end.chokepoint"]
        closure_end_week = observation["closure_end.end_week"]
        closure_obs = observation["closure_end.chokepoint.observed"]
        
        if closure_obs.any():
            for i in np.where(closure_obs)[0]:
                cp_node = int(closure_chokepoint[i])
                end_week = int(closure_end_week[i]) if closure_end_week[i] > 0 else week + 10
                weeks_remaining = max(0, end_week - week)
                if weeks_remaining <= 3:
                    # Boost flow through routes avoiding this chokepoint (routes not using it)
                    # For simplicity, boost all edges near end of closure
                    if weeks_remaining <= 1:
                        boost = 1.3
                    else:
                        boost = 1.15
        
        # Base allocation: send reduced capacity, boost threatened routes
        for slot_idx in range(len(flows)):
            if not action_mask[slot_idx]:
                continue
            edge_idx = self.action_slots["edge"][slot_idx]
            k_idx = self.action_slots["k"][slot_idx]
            
            base_flow = self.capacity[slot_idx] * 0.7
            boost = threatened.get((edge_idx, k_idx), 1.0)
            flows[slot_idx] = base_flow * boost
        
        # Clip to allowed capacity
        allowed = self.capacity * action_mask
        flows = np.minimum(flows, allowed)
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }