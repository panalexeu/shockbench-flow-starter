# 0.23869300481681008
import numpy as np

class Agent:
    """Baseline+: full capacity, but reduce on routes with goods already in flight."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
        
        self.action_slots = action_slots
        self.edges = static["edges"]
        self.sinks = static["sinks"]
    
    def act(self, observation):
        flows = self.capacity * observation["action_mask"]
        
        # Reduce flow on routes with heavy pipeline (goods already in flight)
        pipeline_edge = observation["pipeline.edge"]
        pipeline_k = observation["pipeline.k"]
        pipeline_qty = observation["pipeline.qty"]
        pipeline_observed = observation["pipeline.qty.observed"]
        
        # Count quantity in pipeline per (edge, k) pair
        pipeline_load = {}
        if pipeline_observed.sum() > 0:
            for i in np.where(pipeline_observed)[0]:
                edge_idx = int(pipeline_edge[i])
                k_idx = int(pipeline_k[i])
                qty = float(pipeline_qty[i])
                key = (edge_idx, k_idx)
                pipeline_load[key] = pipeline_load.get(key, 0.0) + qty
        
        # For each action slot, reduce flow if pipeline is heavy
        for slot_idx in range(len(flows)):
            if flows[slot_idx] > 0:
                edge_idx = self.action_slots["edge"][slot_idx]
                k_idx = self.action_slots["k"][slot_idx]
                in_flight = pipeline_load.get((edge_idx, k_idx), 0.0)
                
                # If in-flight quantity exceeds 50% of capacity, reduce new flow by 20%
                if in_flight > 0.5 * self.capacity[slot_idx]:
                    flows[slot_idx] *= 0.8
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }