# 0.23154829213589143
import numpy as np

class Agent:
    """Multi-week lookahead with smooth inventory-aware scaling."""
    
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
        flows = np.zeros_like(self.capacity)
        action_mask = observation["action_mask"]
        
        # Multi-week demand: weight recent weeks heavily
        demand_forecast = observation["demand_forecast.qty"]  # (num_demands, 8)
        backlog = observation["backlog.qty"]
        stock = observation["stock.qty"]
        stock_observed = observation["stock.qty.observed"]
        
        # Compute commodity-level demand across weeks 0-3 with decay
        num_commodities = len(self.commodities["id"])
        demand_signal = np.zeros(num_commodities)
        
        weights = np.array([1.0, 0.95, 0.85, 0.75, 0.6, 0.4, 0.2, 0.1])
        for sink_idx in range(len(self.sinks["k"])):
            k_idx = self.sinks["k"][sink_idx]
            if sink_idx < len(demand_forecast):
                weighted_demand = np.sum(demand_forecast[sink_idx, :] * weights)
                demand_signal[k_idx] += weighted_demand
        
        # Backlog penalty
        for sink_idx in range(len(self.sinks["k"])):
            k_idx = self.sinks["k"][sink_idx]
            if sink_idx < len(backlog):
                demand_signal[k_idx] += backlog[sink_idx] * 3.0
        
        # Stock level modulation: if total stock is high relative to near-term demand, scale down slightly
        total_stock = np.sum(stock[stock_observed == 1]) if stock_observed.sum() > 0 else 0
        total_demand = np.sum(demand_signal)
        
        if total_demand > 0:
            stock_ratio = total_stock / (total_demand + 1e-6)
            # If stock > 2x demand, scale down to 0.85x; if stock < 0.5x demand, scale to 1.0x
            inventory_factor = np.clip(1.0 - 0.15 * max(stock_ratio - 1.0, 0), 0.85, 1.0)
        else:
            inventory_factor = 1.0
        
        # Normalize demand signal
        if total_demand > 0:
            demand_norm = demand_signal / total_demand
        else:
            demand_norm = np.ones(num_commodities) / num_commodities
        
        # Allocate with demand priority and inventory scaling
        for slot_idx in range(len(flows)):
            if action_mask[slot_idx]:
                k_idx = self.action_slots["k"][slot_idx]
                # Base 0.88, boost by demand (up to 0.22)
                demand_factor = 0.88 + 0.22 * min(demand_norm[k_idx] * len(self.commodities["id"]), 1.0)
                flows[slot_idx] = self.capacity[slot_idx] * demand_factor * inventory_factor
        
        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}