# -999
# error: 'c'
import numpy as np

class Agent:
    """Cost-aware routing: prioritize high-value commodities and cheap routes."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        self.action_slots = action_slots
        self.edges = static["edges"]
        self.commodities = static["commodities"]
        self.sinks = static["sinks"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = np.zeros_like(self.capacity)
        action_mask = observation["action_mask"]
        
        # Get demand forecast and backlog
        demand_forecast = observation["demand_forecast.qty"]  # (num_demands, 8)
        backlog = observation["backlog.qty"]
        
        # Compute commodity value and urgency
        num_commodities = len(self.commodities["id"])
        commodity_value = np.array(self.commodities["v"], dtype=float)
        commodity_urgency = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            k = self.sinks["k"][sink_idx]
            # Sum demand forecasts for this commodity
            if sink_idx < len(demand_forecast):
                commodity_urgency[k] += np.sum(demand_forecast[sink_idx, :3])  # 3-week horizon
            # Add backlog
            if sink_idx < len(backlog):
                commodity_urgency[k] += backlog[sink_idx] * 2.0
        
        # Normalize
        max_urgency = np.max(commodity_urgency) if np.max(commodity_urgency) > 0 else 1.0
        commodity_urgency = commodity_urgency / (max_urgency + 1e-9)
        
        # Get route costs
        edge_costs = np.array(self.edges["c"], dtype=float)
        
        # Allocation: weight by urgency and inverse of cost
        for slot_idx in range(len(flows)):
            if not action_mask[slot_idx]:
                continue
            
            edge_idx = self.action_slots["edge"][slot_idx]
            k = self.action_slots["k"][slot_idx]
            
            # Avoid division by zero
            edge_cost = edge_costs[edge_idx] if edge_costs[edge_idx] > 0 else 1.0
            cost_efficiency = 1.0 / (edge_cost + 1e-6)
            
            # Allocate more to high-urgency, low-cost routes
            value_weight = commodity_value[k] / (np.max(commodity_value) + 1e-9)
            urgency_weight = commodity_urgency[k]
            
            # Combined score: urgency + value + cost efficiency
            score = 0.4 * urgency_weight + 0.3 * value_weight + 0.3 * min(cost_efficiency, 1.0)
            
            # Send 0.6 to 1.0 of capacity based on score
            flows[slot_idx] = self.capacity[slot_idx] * (0.6 + 0.4 * score)
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }
