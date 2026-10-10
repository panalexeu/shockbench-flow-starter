# 0.48062062906183634
import numpy as np

class Agent:
    """Demand forecast lookahead: pre-stage goods for anticipated demand spikes."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        layout = config["layout"]
        
        u0 = static["edges"]["u0"]
        tau0 = static["edges"]["tau0"]
        action_slots = static["action_slots"]
        
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        self.edge_tau = np.array(tau0, dtype=float)
        
        self.action_slots = action_slots
        self.commodities = static["commodities"]
        self.layout_demands = layout["demands"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        action_mask = observation["action_mask"]
        flows = self.capacity.copy()
        
        # Get week number to coordinate lead time
        week = int(observation["week"][0])
        
        # Analyze demand forecast
        demand_forecast = observation["demand_forecast.qty"]
        demand_forecast_obs = observation["demand_forecast.qty.observed"]
        
        num_commodities = len(self.commodities["id"])
        
        # Detect demand peaks per commodity in next 8 weeks
        demand_peaks = np.zeros(num_commodities)
        demand_baseline = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.layout_demands)):
            if sink_idx < len(demand_forecast) and demand_forecast_obs[sink_idx].any():
                k_idx = self.layout_demands[sink_idx][1]
                forecast = demand_forecast[sink_idx, :]
                demand_baseline[k_idx] += np.mean(forecast[:3])
                demand_peaks[k_idx] = max(demand_peaks[k_idx], np.max(forecast[2:5]))
        
        # Normalize peak pressure
        max_peak = np.max(demand_peaks) if np.max(demand_peaks) > 0 else 1.0
        peak_pressure = demand_peaks / (max_peak + 1e-9)
        
        # Pre-stage goods for upcoming peaks: send more on short lead-time routes
        for slot_idx in range(len(flows)):
            if not action_mask[slot_idx]:
                flows[slot_idx] = 0
                continue
            
            edge_idx = self.action_slots["edge"][slot_idx]
            k_idx = self.action_slots["k"][slot_idx]
            tau = self.edge_tau[edge_idx]
            
            # If peak is forecasted and this edge's lead time can catch it, boost
            if peak_pressure[k_idx] > 0.3:
                if tau <= 4:  # Can reach demand in time
                    boost = 1.0 + 0.25 * peak_pressure[k_idx]
                else:
                    boost = 0.95  # Long lead-time, may overshoot
            else:
                # Baseline demand: send normally
                boost = 1.0
            
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