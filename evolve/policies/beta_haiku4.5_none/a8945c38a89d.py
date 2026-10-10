# 0.47151635098253686
import numpy as np

class Agent:
    """Short-lead-time routing: prefer edges with tau <= 2 weeks, boost them when pipeline low."""
    
    def __init__(self, config=None):
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        u0 = static["edges"]["u0"]
        tau0 = static["edges"]["tau0"]
        action_slots = static["action_slots"]
        
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        self.edge_tau = np.array(tau0, dtype=float)
        
        self.action_slots = action_slots
        self.edges = static["edges"]
        self.commodities = static["commodities"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = self.capacity.copy()
        action_mask = observation["action_mask"]
        
        # Compute pipeline per commodity
        pipeline_k = observation["pipeline.k"]
        pipeline_qty = observation["pipeline.qty"]
        pipeline_obs = observation["pipeline.qty.observed"]
        
        num_commodities = len(self.commodities["id"])
        pipeline_per_k = np.zeros(num_commodities)
        
        for i in np.where(pipeline_obs)[0]:
            k_idx = int(pipeline_k[i])
            if k_idx < num_commodities:
                pipeline_per_k[k_idx] += pipeline_qty[i]
        
        max_pipeline = np.max(pipeline_per_k) if np.max(pipeline_per_k) > 0 else 1.0
        pipeline_norm = pipeline_per_k / (max_pipeline + 1e-9)
        
        # Boost per slot: high for short lead time + low pipeline
        for slot_idx in range(len(flows)):
            edge_idx = self.action_slots["edge"][slot_idx]
            k_idx = self.action_slots["k"][slot_idx]
            
            tau = self.edge_tau[edge_idx]
            
            # Short lead-time boost
            if tau <= 2:
                lead_boost = 1.3
            elif tau <= 4:
                lead_boost = 1.1
            else:
                lead_boost = 0.9
            
            # Low-pipeline boost
            pipeline_boost = 1.0 + 0.25 * (1.0 - pipeline_norm[k_idx])
            
            flows[slot_idx] *= lead_boost * pipeline_boost
        
        flows = flows * action_mask
        
        # Soft cap
        allowed_total = np.sum(self.capacity * action_mask)
        flows_total = np.sum(flows)
        if flows_total > allowed_total * 1.25:
            flows = flows * (allowed_total * 1.2 / (flows_total + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }