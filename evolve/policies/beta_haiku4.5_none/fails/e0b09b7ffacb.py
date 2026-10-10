# -999
# error: 832 of 832 weeks crashed, first: episode 0, week 0: KeyError: 'supply_slots'
import numpy as np

class Agent:
    """Forecast-aware: anticipate demand and build stock preemptively."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
        
        # Map edges to sink demands for better targeting
        self.edges = static["edges"]
        self.sinks = static["sinks"]
        self.supply_slots = static["supply_slots"]
        self.action_slots = action_slots
        self.rng = np.random.default_rng(config["policy_seed"])
    
    def act(self, observation):
        flows = np.zeros_like(self.capacity)
        week = int(observation["week"][0])
        
        # Get demand forecast (up to 8 weeks ahead)
        demand_forecast = observation["demand_forecast.qty"]  # shape (num_demands, 8)
        demand_observed = observation["demand_forecast.qty.observed"]
        stock = observation["stock.qty"]
        stock_observed = observation["stock.qty.observed"]
        
        # Simple strategy: allocate flows proportionally to forecast demand
        # Prioritize slots that feed high-demand paths
        total_forecast = np.sum(demand_forecast[:, 0]) if demand_observed.sum() > 0 else 1.0
        
        if total_forecast > 0:
            # Compute demand weight per commodity
            demand_weight = np.zeros(self.config["static"]["commodities"]["id"].__len__())
            for i, d in enumerate(demand_forecast):
                if demand_observed[i]:
                    demand_weight[self.sinks["k"][i]] += d[0]
            
            # Allocate capacity proportionally, respecting masks and capacity
            for slot_idx in range(len(flows)):
                if observation["action_mask"][slot_idx]:
                    k = self.action_slots["k"][slot_idx]
                    if demand_weight[k] > 0:
                        # Send up to capacity for high-demand commodities
                        flows[slot_idx] = self.capacity[slot_idx]
                    else:
                        # Send at reduced rate for low-demand commodities
                        flows[slot_idx] = self.capacity[slot_idx] * 0.3
        
        # Fallback: use capacity mask if forecast unavailable
        if flows.sum() == 0:
            flows = self.capacity * observation["action_mask"]
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }