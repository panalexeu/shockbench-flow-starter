# 0.18479283728398604
import numpy as np

class Agent:
    """Prohibition and tariff aware with dynamic allocation."""
    
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
        self.edges = static["edges"]
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = self.capacity.copy()
        action_mask = observation["action_mask"]
        flows = flows * action_mask
        
        # Backlog and demand per commodity
        backlog = observation["backlog.qty"]
        demand_forecast = observation["demand_forecast.qty"]
        num_commodities = len(self.commodities["id"])
        
        urgency_per_k = np.zeros(num_commodities)
        for sink_idx in range(len(self.sinks["k"])):
            k_idx = self.sinks["k"][sink_idx]
            if sink_idx < len(backlog):
                urgency_per_k[k_idx] += backlog[sink_idx] * 2.0
            if sink_idx < demand_forecast.shape[0]:
                urgency_per_k[k_idx] += demand_forecast[sink_idx, 0] * 1.0 + np.sum(demand_forecast[sink_idx, 1:3]) * 0.5
        
        max_urgency = np.max(urgency_per_k) if np.max(urgency_per_k) > 1e-6 else 1.0
        urgency_norm = urgency_per_k / (max_urgency + 1e-9)
        
        # Tariff penalty
        tariff = observation["graph_now.tariff"]
        tariff_obs = observation["graph_now.tariff.observed"]
        mean_tariff = np.mean(tariff[tariff_obs.astype(bool)]) if tariff_obs.sum() > 0 else 1.0
        
        # Prohibited edges
        prohibited = observation["graph_now.prohibited"]
        
        for slot_idx in range(len(flows)):
            if not action_mask[slot_idx]:
                flows[slot_idx] = 0.0
                continue
            
            edge_idx = self.action_slots["edge"][slot_idx]
            k_idx = self.action_slots["k"][slot_idx]
            
            # Check prohibition
            if prohibited[edge_idx, k_idx] > 0:
                flows[slot_idx] = 0.0
                continue
            
            # Base allocation from urgency
            allocation_factor = 0.75 + 0.30 * urgency_norm[k_idx]
            
            # Tariff penalty (reduce on high-tariff routes)
            if tariff_obs[edge_idx, k_idx]:
                route_tariff = tariff[edge_idx, k_idx]
                if route_tariff > 1.4 * mean_tariff:
                    allocation_factor *= 0.85
            
            flows[slot_idx] *= allocation_factor
        
        # Soft cap
        allowed_capacity = self.capacity * action_mask
        total_allowed = np.sum(allowed_capacity)
        total_flows = np.sum(flows)
        if total_flows > total_allowed * 1.2:
            flows = flows * (total_allowed * 1.15 / (total_flows + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }