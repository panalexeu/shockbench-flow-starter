# 0.4699062375097168
import numpy as np

class Agent:
    """Lead-time routing with pipeline balancing."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        u0 = static["edges"]["u0"]
        tau0 = static["edges"]["tau0"]
        c0 = static["edges"]["c0"]
        action_slots = static["action_slots"]
        
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        self.edge_tau = np.array(tau0, dtype=float)
        self.edge_cost = np.array([c0[e] if c0[e] is not None else 1e6 for e in action_slots["edge"]], dtype=float)
        
        self.action_slots = action_slots
        self.commodities = static["commodities"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        action_mask = observation["action_mask"]
        flows = self.capacity.copy()
        
        # Compute pipeline depth per commodity
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
        
        # Apply routing boost
        for slot_idx in range(len(flows)):
            if not action_mask[slot_idx]:
                flows[slot_idx] = 0
                continue
            
            edge_idx = self.action_slots["edge"][slot_idx]
            k_idx = self.action_slots["k"][slot_idx]
            tau = self.edge_tau[edge_idx]
            
            # Boost short lead-time routes more when pipeline is high
            pipeline_pressure = pipeline_norm[k_idx]
            
            if tau <= 2:
                boost = 1.25
            elif tau <= 4:
                boost = 1.10
            else:
                # Long lead-time: boost only when pipeline is low
                boost = 0.9 + 0.3 * (1.0 - pipeline_pressure)
            
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