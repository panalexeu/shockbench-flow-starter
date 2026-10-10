# 0.32414970511382535
import numpy as np

class Agent:
    """Demand-forecast-driven: safe multi-week lookahead without array ambiguity."""
    
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
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = self.capacity.copy()
        action_mask = observation["action_mask"]
        
        # Get demand forecast (8 weeks ahead)
        demand_forecast = observation["demand_forecast.qty"]  # (num_demands, 8)
        backlog = observation["backlog.qty"]
        
        # Compute weighted demand signal per commodity: focus on near-term
        num_commodities = len(self.commodities["id"])
        weighted_demand = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            k_idx = self.sinks["k"][sink_idx]
            if sink_idx < demand_forecast.shape[0]:
                # Weight: current + next 2 weeks
                d0 = demand_forecast[sink_idx, 0]
                d1 = demand_forecast[sink_idx, 1] if demand_forecast.shape[1] > 1 else 0.0
                d2 = demand_forecast[sink_idx, 2] if demand_forecast.shape[1] > 2 else 0.0
                weighted_demand[k_idx] += d0 + 0.8 * d1 + 0.6 * d2
        
        # Add backlog signal
        backlog_signal = np.zeros(num_commodities)
        for sink_idx in range(len(self.sinks["k"])):
            k_idx = self.sinks["k"][sink_idx]
            if sink_idx < len(backlog):
                backlog_signal[k_idx] += backlog[sink_idx]
        
        # Total urgency
        urgency = weighted_demand + backlog_signal * 1.5
        max_urgency = np.max(urgency)
        if max_urgency > 1e-6:
            urgency_normalized = urgency / (max_urgency + 1e-9)
        else:
            urgency_normalized = np.ones(num_commodities) / num_commodities
        
        # Allocate: base 0.85x + up to 0.30x boost based on urgency
        flows = flows * action_mask
        for slot_idx in range(len(flows)):
            k_idx = self.action_slots["k"][slot_idx]
            allocation_factor = 0.85 + 0.30 * urgency_normalized[k_idx]
            flows[slot_idx] *= allocation_factor
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }