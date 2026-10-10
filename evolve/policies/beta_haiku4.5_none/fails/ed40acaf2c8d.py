# -999
# error: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
import numpy as np

class Agent:
    """Demand-forecast-driven with multi-week lookahead."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        layout = config["layout"]
        
        # Extract nominal capacities
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        self.action_slots = action_slots
        self.edges = static["edges"]
        self.sinks = static["sinks"]
        self.commodities = static["commodities"]
        self.layout = layout
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = np.zeros_like(self.capacity)
        action_mask = observation["action_mask"]
        
        # Get demand forecast (8 weeks ahead)
        demand_forecast = observation["demand_forecast.qty"]  # (num_demands, 8)
        demand_observed = observation["demand_forecast.qty.observed"]
        backlog = observation["backlog.qty"]
        stock = observation["stock.qty"]
        stock_observed = observation["stock.qty.observed"]
        
        # Compute commodity-level demand signals
        num_commodities = len(self.commodities["id"])
        demand_signal = np.zeros(num_commodities)
        
        if demand_observed.sum() > 0:
            for sink_idx, (d_row, is_obs) in enumerate(zip(demand_forecast, demand_observed)):
                if is_obs:
                    k_idx = self.sinks["k"][sink_idx]
                    # Weight: current + next 2 weeks heavily, then decay
                    weighted_demand = d_row[0] + 0.9 * d_row[1] + 0.7 * d_row[2]
                    demand_signal[k_idx] += weighted_demand
        
        # Compute backlog pressure per commodity
        backlog_signal = np.zeros(num_commodities)
        for sink_idx in range(len(self.sinks["k"])):
            k_idx = self.sinks["k"][sink_idx]
            if sink_idx < len(backlog):
                backlog_signal[k_idx] += backlog[sink_idx]
        
        # Total urgency: demand + backlog penalty
        urgency = demand_signal + backlog_signal * 2.0
        total_urgency = np.sum(urgency) + 1e-6
        
        # Allocate capacity proportional to urgency, respecting masks
        for slot_idx in range(len(flows)):
            if action_mask[slot_idx]:
                k_idx = self.action_slots["k"][slot_idx]
                if total_urgency > 0:
                    urgency_weight = urgency[k_idx] / total_urgency
                    # Ramp: base 0.8x + up to 1.2x based on urgency
                    allocation_factor = 0.8 + 0.4 * min(urgency_weight * 10, 1.0)
                else:
                    allocation_factor = 0.8
                flows[slot_idx] = self.capacity[slot_idx] * allocation_factor
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }
