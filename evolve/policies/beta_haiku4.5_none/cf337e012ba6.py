# 0.28039523901183616
import numpy as np

class Agent:
    """Lead-time aware with dynamic pressure modulation."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        u0 = static["edges"]["u0"]
        tau0 = static["edges"]["tau0"]
        action_slots = static["action_slots"]
        
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        self.edge_tau = np.array(tau0, dtype=float)
        
        self.action_slots = action_slots
        self.sinks = static["sinks"]
        self.commodities = static["commodities"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = self.capacity.copy()
        action_mask = observation["action_mask"]
        
        num_commodities = len(self.commodities["id"])
        
        # Pressure signal: backlog + demand
        backlog = observation["backlog.qty"]
        backlog_per_k = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(backlog):
                k_idx = self.sinks["k"][sink_idx]
                backlog_per_k[k_idx] += backlog[sink_idx]
        
        demand_forecast = observation["demand_forecast.qty"]
        demand_per_k = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < demand_forecast.shape[0]:
                k_idx = self.sinks["k"][sink_idx]
                demand_per_k[k_idx] += demand_forecast[sink_idx, 0] + 0.9 * demand_forecast[sink_idx, 1]
        
        max_backlog = np.max(backlog_per_k) if np.max(backlog_per_k) > 0 else 1.0
        max_demand = np.max(demand_per_k) if np.max(demand_per_k) > 0 else 1.0
        
        pressure = 0.65 * (backlog_per_k / (max_backlog + 1e-9)) + 0.35 * (demand_per_k / (max_demand + 1e-9))
        
        # Lead-time boost: long routes need early loading
        for slot_idx in range(len(flows)):
            if action_mask[slot_idx]:
                k_idx = self.action_slots["k"][slot_idx]
                edge_idx = self.action_slots["edge"][slot_idx]
                tau = self.edge_tau[edge_idx]
                
                # Base: 0.78 + 0.22 * pressure
                intensity = 0.78 + 0.22 * min(pressure[k_idx], 1.0)
                
                # Lead-time boost: more aggressive on long routes
                if tau > 3:
                    lead_boost = 1.12
                elif tau > 1:
                    lead_boost = 1.06
                else:
                    lead_boost = 1.02
                
                flows[slot_idx] *= intensity * lead_boost
            else:
                flows[slot_idx] = 0.0
        
        # Cap
        allowed = np.sum(self.capacity * action_mask)
        total = np.sum(flows)
        if total > allowed * 1.3:
            flows *= (allowed * 1.25 / (total + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }