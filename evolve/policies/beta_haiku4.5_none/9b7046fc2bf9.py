# -0.18675059560346638
import numpy as np

class Agent:
    """Stock-aware: balance capacity sending with stock levels to smooth flow."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        # Extract nominal capacities
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        # Store structure
        self.action_slots = action_slots
        self.edges = static["edges"]
        self.commodities = static["commodities"]
        self.sinks = static["sinks"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = np.zeros_like(self.capacity)
        action_mask = observation["action_mask"]
        stock = observation["stock.qty"]
        stock_observed = observation["stock.qty.observed"]
        demand_forecast = observation["demand_forecast.qty"]
        backlog = observation["backlog.qty"]
        
        week = int(observation["week"][0])
        
        # Compute total stock and backlog
        total_stock = np.sum(stock) if stock_observed.any() else 0
        total_backlog = np.sum(backlog)
        
        # Compute next week's forecast demand
        next_demand = np.sum(demand_forecast[:, 1]) if observation["demand_forecast.qty.observed"].any() else 1e6
        
        # Throttle if we have ample stock; ramp up if backlog is high
        if total_stock > next_demand * 2:
            # High inventory: send less to avoid overproduction
            allocation_factor = 0.5
        elif total_backlog > next_demand * 0.5:
            # High backlog: send aggressively
            allocation_factor = 1.2
        else:
            # Normal: send at capacity
            allocation_factor = 1.0
        
        flows = self.capacity * action_mask * allocation_factor
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }