# -0.014771322157051193
import numpy as np

class Agent:
    """Pipeline-congestion-aware: throttle when routes have heavy in-flight load."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        self.action_slots = action_slots
        self.edges = static["edges"]
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = self.capacity.copy()
        action_mask = observation["action_mask"]
        
        # Track in-flight quantities by (edge, commodity)
        pipeline_edge = observation["pipeline.edge"]
        pipeline_k = observation["pipeline.k"]
        pipeline_qty = observation["pipeline.qty"]
        pipeline_observed = observation["pipeline.qty.observed"]
        
        in_flight = {}
        if pipeline_observed.sum() > 0:
            for i in np.where(pipeline_observed)[0]:
                edge_idx = int(pipeline_edge[i])
                k_idx = int(pipeline_k[i])
                qty = float(pipeline_qty[i])
                key = (edge_idx, k_idx)
                in_flight[key] = in_flight.get(key, 0.0) + qty
        
        # Adjust flows based on pipeline load
        for slot_idx in range(len(flows)):
            if not action_mask[slot_idx]:
                flows[slot_idx] = 0.0
                continue
            
            edge_idx = self.action_slots["edge"][slot_idx]
            k_idx = self.action_slots["k"][slot_idx]
            key = (edge_idx, k_idx)
            
            cap = self.capacity[slot_idx]
            in_flight_qty = in_flight.get(key, 0.0)
            
            # Pipeline load ratio
            load_ratio = in_flight_qty / (cap + 1e-6) if cap > 0 else 0.0
            
            # Throttle curve: full at low load, reduces as load increases
            # At load_ratio=1.0 (capacity in flight), send 0.5x capacity
            # At load_ratio=2.0 (2x capacity in flight), send 0.2x capacity
            if load_ratio < 0.5:
                factor = 1.0
            elif load_ratio < 1.0:
                factor = 1.0 - 0.5 * (load_ratio - 0.5) / 0.5  # 1.0 to 0.75
            elif load_ratio < 2.0:
                factor = 0.75 - 0.55 * (load_ratio - 1.0) / 1.0  # 0.75 to 0.2
            else:
                factor = 0.2
            
            flows[slot_idx] = cap * factor
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }
