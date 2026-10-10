# 0.25713569492265304
import numpy as np

class Agent:
    """Demand-driven with multi-horizon weighting and pipeline awareness."""
    
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
        
        # Multi-horizon demand signal
        demand_forecast = observation["demand_forecast.qty"]
        num_commodities = len(self.commodities["id"])
        weighted_demand_per_k = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < demand_forecast.shape[0]:
                k_idx = self.sinks["k"][sink_idx]
                # Exponential decay weights: 1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05
                weights = np.array([1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05])
                for h in range(min(8, demand_forecast.shape[1])):
                    weighted_demand_per_k[k_idx] += demand_forecast[sink_idx, h] * weights[h]
        
        # Backlog signal
        backlog = observation["backlog.qty"]
        backlog_per_k = np.zeros(num_commodities)
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(backlog):
                k_idx = self.sinks["k"][sink_idx]
                backlog_per_k[k_idx] += backlog[sink_idx]
        
        # Pipeline inventory (in-transit stock)
        pipeline_k = observation["pipeline.k"]
        pipeline_qty = observation["pipeline.qty"]
        pipeline_observed = observation["pipeline.qty.observed"]
        pipeline_per_k = np.zeros(num_commodities)
        
        if pipeline_observed.any():
            for i in range(len(pipeline_k)):
                if pipeline_observed[i]:
                    k_idx = pipeline_k[i]
                    if k_idx < num_commodities:
                        pipeline_per_k[k_idx] += pipeline_qty[i]
        
        # Stock inventory
        stock_qty = observation["stock.qty"]
        stock_k_list = self.config["layout"]["stock_slots"]  # [(node, k), ...]
        stock_per_k = np.zeros(num_commodities)
        for idx, (node, k) in enumerate(stock_k_list):
            if idx < len(stock_qty) and k < num_commodities:
                stock_per_k[k] += stock_qty[idx]
        
        # Net urgency: demand - inventory
        total_inventory_per_k = stock_per_k + pipeline_per_k
        net_urgency = weighted_demand_per_k + backlog_per_k - 0.5 * total_inventory_per_k
        net_urgency = np.maximum(net_urgency, 0)  # Non-negative urgency
        
        max_urgency = np.max(net_urgency) if np.max(net_urgency) > 1e-6 else 1.0
        urgency_normalized = net_urgency / (max_urgency + 1e-9)
        
        # Apply allocation factor: base 0.80 + up to 0.35 boost
        for slot_idx in range(len(flows)):
            k_idx = self.action_slots["k"][slot_idx]
            allocation_factor = 0.80 + 0.35 * urgency_normalized[k_idx]
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