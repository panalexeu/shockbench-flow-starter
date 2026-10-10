# 0.00258045890229412
import numpy as np

class Agent:
    """Cost-aware with slot-level tariff and freight filtering."""
    
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
        self.sinks = static["sinks"]
        self.commodities = static["commodities"]
    
    def act(self, observation):
        flows = self.capacity * observation["action_mask"]
        action_mask = observation["action_mask"]
        
        # Demand signal
        backlog = observation["backlog.qty"]
        backlog_obs = observation["backlog.qty.observed"]
        demand_forecast = observation["demand_forecast.qty"]
        demand_forecast_obs = observation["demand_forecast.qty.observed"]
        
        num_commodities = len(self.commodities["id"])
        demand_signal = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            k_idx = self.sinks["k"][sink_idx]
            if sink_idx < len(backlog_obs) and backlog_obs[sink_idx] > 0:
                demand_signal[k_idx] += backlog[sink_idx]
            if sink_idx < len(demand_forecast_obs) and demand_forecast_obs[sink_idx].sum() > 0:
                demand_signal[k_idx] += demand_forecast[sink_idx, 0]
        
        max_demand = np.max(demand_signal) if np.max(demand_signal) > 0 else 1.0
        demand_norm = demand_signal / (max_demand + 1e-9)
        
        # Cost data
        graph_c = observation["graph_now.c"]
        graph_tariff = observation["graph_now.tariff"]
        tariff_obs = observation["graph_now.tariff.observed"]
        
        cost_p75 = np.percentile(graph_c[graph_c > 0], 75) if (graph_c > 0).sum() > 0 else 1e6
        cost_p50 = np.median(graph_c[graph_c > 0]) if (graph_c > 0).sum() > 0 else 1.0
        
        # Per-slot modulation
        for slot_idx in range(len(flows)):
            if action_mask[slot_idx]:
                k_idx = self.action_slots["k"][slot_idx]
                edge_idx = self.action_slots["edge"][slot_idx]
                
                # Base from demand
                base_factor = 0.7 + 0.3 * demand_norm[k_idx]
                
                # Freight cost reduction
                cost_factor = 1.0
                if edge_idx < len(graph_c):
                    edge_cost = graph_c[edge_idx]
                    if edge_cost > cost_p75:
                        cost_factor *= 0.88
                    elif edge_cost > cost_p50 * 1.2:
                        cost_factor *= 0.94
                
                # Tariff reduction
                if tariff_obs.sum() > 0 and edge_idx < len(graph_tariff) and k_idx < len(graph_tariff[0]):
                    tariff_val = graph_tariff[edge_idx, k_idx]
                    if tariff_val > 0.20:
                        cost_factor *= 0.75
                    elif tariff_val > 0.10:
                        cost_factor *= 0.88
                    elif tariff_val > 0.05:
                        cost_factor *= 0.95
                
                flows[slot_idx] *= base_factor * cost_factor
        
        # Cap
        allowed_total = np.sum(self.capacity * action_mask)
        flows_total = np.sum(flows)
        if flows_total > allowed_total * 1.05:
            flows = flows * (allowed_total / (flows_total + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }