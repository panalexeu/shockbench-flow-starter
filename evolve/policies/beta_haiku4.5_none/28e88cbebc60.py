# 0.22955750262253025
import numpy as np

class Agent:
    """Conservative lead-time aware: boost long routes moderately based on pressure."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        u0 = static["edges"]["u0"]
        tau0 = static["edges"]["tau0"]
        action_slots = static["action_slots"]
        
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        self.edge_tau = np.array(tau0, dtype=float)
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
        
        self.action_slots = action_slots
        self.sinks = static["sinks"]
        self.commodities = static["commodities"]
    
    def act(self, observation):
        flows = self.capacity.copy()
        action_mask = observation["action_mask"]
        
        num_commodities = len(self.commodities["id"])
        
        # Compute pressure from backlog + next 2 weeks of demand
        backlog = observation["backlog.qty"]
        backlog_obs = observation["backlog.qty.observed"]
        demand_forecast = observation["demand_forecast.qty"]
        demand_forecast_obs = observation["demand_forecast.qty.observed"]
        
        pressure = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            k_idx = self.sinks["k"][sink_idx]
            if sink_idx < len(backlog_obs) and backlog_obs[sink_idx] > 0:
                pressure[k_idx] += backlog[sink_idx] * 1.5
            if sink_idx < len(demand_forecast_obs) and demand_forecast_obs[sink_idx].sum() > 0:
                # Current + next week with decay
                pressure[k_idx] += demand_forecast[sink_idx, 0] + 0.7 * demand_forecast[sink_idx, 1]
        
        max_pressure = np.max(pressure) if np.max(pressure) > 0 else 1.0
        pressure_norm = pressure / (max_pressure + 1e-9)
        
        # Apply modulation with lead-time boost
        for slot_idx in range(len(flows)):
            if action_mask[slot_idx]:
                k_idx = self.action_slots["k"][slot_idx]
                edge_idx = self.action_slots["edge"][slot_idx]
                tau = self.edge_tau[edge_idx]
                
                # Base intensity from pressure
                intensity = 0.75 + 0.25 * pressure_norm[k_idx]
                
                # Lead-time boost: longer routes get modest boost
                if tau >= 4:
                    lead_boost = 1.08
                elif tau >= 2:
                    lead_boost = 1.04
                else:
                    lead_boost = 1.0
                
                flows[slot_idx] *= intensity * lead_boost
            else:
                flows[slot_idx] = 0.0
        
        # Cap to capacity
        allowed_total = np.sum(self.capacity * action_mask)
        flows_total = np.sum(flows)
        if flows_total > allowed_total * 1.2:
            flows = flows * (allowed_total * 1.15 / (flows_total + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }