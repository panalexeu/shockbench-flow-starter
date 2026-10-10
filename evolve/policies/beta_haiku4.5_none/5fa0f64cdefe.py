# 0.38694824991214144
import numpy as np

class Agent:
    """Demand-driven with minimal intervention: stay close to capacity on allowed slots."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        self.action_slots = action_slots
        self.commodities = static["commodities"]
        self.layout_demands = config["layout"]["demands"]
        self.override_slots = static["override_slots"]
        self.layout_chokepoints = config["layout"]["chokepoints"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        # Base: full capacity where allowed
        flows = self.capacity * observation["action_mask"]
        
        # Mild demand-aware scaling: reduce underutilized slots
        backlog = observation["backlog.qty"]
        demand_forecast = observation["demand_forecast.qty"]
        num_demands = len(self.layout_demands)
        
        # Compute average demand per sink
        avg_demand_per_sink = np.zeros(num_demands)
        for sink_idx in range(num_demands):
            if sink_idx < len(demand_forecast):
                avg_demand_per_sink[sink_idx] = np.mean(demand_forecast[sink_idx, :4])
        
        # Map slots to sink demand
        total_flow_per_k = np.zeros(len(self.commodities["id"]))
        for slot_idx in range(len(flows)):
            k = self.action_slots["k"][slot_idx]
            total_flow_per_k[k] += flows[slot_idx]
        
        # Check if we're over-sending any commodity
        for slot_idx in range(len(flows)):
            k = self.action_slots["k"][slot_idx]
            
            # Find max demand for this commodity
            max_sink_demand = 0.0
            for sink_idx in range(num_demands):
                if self.layout_demands[sink_idx][1] == k:
                    max_sink_demand = max(max_sink_demand, avg_demand_per_sink[sink_idx])
            
            # If total flow >> demand, scale down slightly
            if total_flow_per_k[k] > max(10.0, 3.0 * max_sink_demand):
                flows[slot_idx] *= 0.9
        
        # Override logic: minimal—only for severe constraints
        graph_open = observation["graph_now.open"]
        override_mask = observation["override_mask"]
        
        for override_idx in range(len(self.override_slots["chokepoint"])):
            if not override_mask[override_idx]:
                continue
            
            cp_idx = self.override_slots["chokepoint"][override_idx]
            if cp_idx < len(graph_open) and graph_open[cp_idx] < 0.3:
                # Only use override on nearly-closed chokepoints
                self.release_mode[override_idx] = 1
                self.override_qty[override_idx] = 30.0
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }