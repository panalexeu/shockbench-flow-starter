# -999
# error: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
import numpy as np

class Agent:
    """Policy 1: Conservative demand-driven allocation."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        layout = config["layout"]
        
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        self.action_slots = action_slots
        self.layout_demands = layout["demands"]
        self.sinks = static["sinks"]
        self.edges = static["edges"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = np.zeros_like(self.capacity)
        action_mask = observation["action_mask"]
        
        # Compute demand urgency per commodity
        demand_forecast = observation["demand_forecast.qty"]
        backlog = observation["backlog.qty"]
        demand_forecast_obs = observation["demand_forecast.qty.observed"]
        backlog_obs = observation["backlog.qty.observed"]
        
        num_commodities = len(self.sinks["k"])
        commodity_demand = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.layout_demands)):
            k = self.sinks["k"][sink_idx]
            # Backlog is 2x urgent
            if backlog_obs[sink_idx]:
                commodity_demand[k] += backlog[sink_idx] * 2.0
            # Current and near-term forecast
            if demand_forecast_obs[sink_idx]:
                commodity_demand[k] += np.sum(demand_forecast[sink_idx, :3])
        
        # Normalize demand
        max_demand = np.max(commodity_demand) if np.max(commodity_demand) > 0 else 1.0
        commodity_demand = commodity_demand / (max_demand + 1e-9)
        
        # Allocate to slots proportional to commodity demand
        for slot_idx in range(len(flows)):
            if not action_mask[slot_idx]:
                continue
            k = self.action_slots["k"][slot_idx]
            # Send 0.5 to 1.0 of capacity, scaled by demand
            demand_factor = 0.5 + 0.5 * commodity_demand[k]
            flows[slot_idx] = self.capacity[slot_idx] * demand_factor
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }