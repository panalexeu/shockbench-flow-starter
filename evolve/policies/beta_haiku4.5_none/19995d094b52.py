# 0.10051967062161399
import numpy as np

class Agent:
    """Enhanced baseline: cost reduction + demand-driven modulation."""
    
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
        self.commodities = static["commodities"]
    
    def act(self, observation):
        flows = self.capacity * observation["action_mask"]
        
        num_commodities = len(self.commodities["id"])
        
        # Compute urgency per commodity: backlog + near-term forecast
        backlog = observation["backlog.qty"]
        backlog_obs = observation["backlog.qty.observed"]
        demand_forecast = observation["demand_forecast.qty"]
        demand_forecast_obs = observation["demand_forecast.qty.observed"]
        
        urgency = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            k_idx = self.sinks["k"][sink_idx]
            if sink_idx < len(backlog_obs):
                if backlog_obs[sink_idx] > 0:
                    urgency[k_idx] += backlog[sink_idx] * 2.0  # Backlog priority
            if sink_idx < len(demand_forecast_obs):
                if demand_forecast_obs[sink_idx].sum() > 0:
                    urgency[k_idx] += demand_forecast[sink_idx, 0]  # Week 0 demand
        
        # Normalize urgency
        max_urgency = np.max(urgency) if np.max(urgency) > 0 else 1.0
        urgency_norm = urgency / (max_urgency + 1e-9)
        
        # Cost modulation: reduce on expensive/high-tariff routes
        graph_c = observation["graph_now.c"]
        graph_tariff = observation["graph_now.tariff"]
        tariff_obs = observation["graph_now.tariff.observed"]
        
        high_cost_threshold = np.percentile(graph_c[graph_c > 0], 75) if (graph_c > 0).sum() > 0 else 1e6
        
        for slot_idx in range(len(flows)):
            if flows[slot_idx] > 0:
                edge_idx = self.action_slots["edge"][slot_idx]
                k_idx = self.action_slots["k"][slot_idx]
                
                cost_factor = 1.0
                
                # High freight cost penalty
                if edge_idx < len(graph_c) and graph_c[edge_idx] > high_cost_threshold:
                    cost_factor *= 0.85
                
                # Tariff penalty
                if tariff_obs.sum() > 0 and edge_idx < len(graph_tariff) and k_idx < len(graph_tariff[0]):
                    tariff_val = graph_tariff[edge_idx, k_idx]
                    if tariff_val > 0.15:
                        cost_factor *= max(0.8, 1.0 - tariff_val * 0.4)
                
                # Apply urgency boost
                urgency_boost = 0.8 + 0.4 * urgency_norm[k_idx]
                flows[slot_idx] *= urgency_boost * cost_factor
        
        # Soft cap to capacity
        allowed_total = np.sum(self.capacity * observation["action_mask"])
        flows_total = np.sum(flows)
        if flows_total > allowed_total * 1.1:
            flows = flows * (allowed_total / (flows_total + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }