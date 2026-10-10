# 0.41915838448004816
import numpy as np

class Agent:
    """Cost-aware routing with tariff penalties."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        u0 = static["edges"]["u0"]
        c0 = static["edges"]["c0"]
        action_slots = static["action_slots"]
        
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        self.edge_cost = np.array([c0[e] if c0[e] is not None else 1e6 for e in action_slots["edge"]], dtype=float)
        
        self.action_slots = action_slots
        self.commodities = static["commodities"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        action_mask = observation["action_mask"]
        flows = self.capacity.copy()
        
        # Get tariff rates
        tariff = observation["graph_now.tariff"]  # (num_edges, num_commodities)
        tariff_obs = observation["graph_now.tariff.observed"]
        
        # Compute mean tariff to identify expensive routes
        if tariff_obs.any():
            tariff_mask = tariff_obs.astype(bool)
            mean_tariff = np.mean(tariff[tariff_mask])
        else:
            mean_tariff = 1.0
        
        # Apply cost penalty
        for slot_idx in range(len(flows)):
            if not action_mask[slot_idx]:
                flows[slot_idx] = 0
                continue
            
            edge_idx = self.action_slots["edge"][slot_idx]
            k_idx = self.action_slots["k"][slot_idx]
            
            # Base: send most capacity
            send_fraction = 0.95
            
            # Apply tariff penalty only if route is expensive
            if tariff_obs[edge_idx, k_idx]:
                route_tariff = tariff[edge_idx, k_idx]
                if route_tariff > 1.3 * mean_tariff:
                    # Reduce on expensive routes, but maintain floor
                    send_fraction = 0.70
                elif route_tariff > mean_tariff:
                    send_fraction = 0.85
            
            flows[slot_idx] *= send_fraction
        
        flows = flows * action_mask
        
        # Soft cap
        allowed_total = np.sum(self.capacity * action_mask)
        flows_total = np.sum(flows)
        if flows_total > allowed_total * 1.1:
            flows = flows * (allowed_total * 1.05 / (flows_total + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }