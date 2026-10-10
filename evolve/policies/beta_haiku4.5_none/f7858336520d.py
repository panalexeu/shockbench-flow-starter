# 0.3173898248481889
import numpy as np

class Agent:
    """Baseline+: full capacity, reduced on expensive/risky routes."""
    
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
    
    def act(self, observation):
        flows = self.capacity * observation["action_mask"]
        
        # Get current tariffs and war risk
        graph_tariff = observation["graph_now.tariff"]  # (num_edges, num_commodities)
        graph_war_risk = observation["graph_now.war_risk"]  # (num_chokepoints,)
        graph_c = observation["graph_now.c"]  # (num_edges,) freight cost
        tariff_observed = observation["graph_now.tariff.observed"]
        
        # Identify expensive routes (high tariff or high freight cost)
        high_tariff_threshold_pct = 0.15  # 15% tariff
        high_cost_threshold = np.percentile(graph_c, 75) if len(graph_c) > 0 else 1e6
        
        for slot_idx in range(len(flows)):
            if flows[slot_idx] == 0:
                continue
            
            edge_idx = self.action_slots["edge"][slot_idx]
            k_idx = self.action_slots["k"][slot_idx]
            
            cost_factor = 1.0
            
            # Check freight cost
            if edge_idx < len(graph_c):
                edge_cost = float(graph_c[edge_idx])
                if edge_cost > high_cost_threshold:
                    cost_factor *= 0.85  # Reduce by 15% on expensive routes
            
            # Check tariff
            if tariff_observed.sum() > 0 and edge_idx < len(graph_c) and k_idx < 8:
                tariff = float(graph_tariff[edge_idx, k_idx])
                if tariff > high_tariff_threshold_pct:
                    cost_factor *= (1.0 - min(0.2, tariff))  # Reduce proportionally to tariff
            
            flows[slot_idx] *= cost_factor
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }