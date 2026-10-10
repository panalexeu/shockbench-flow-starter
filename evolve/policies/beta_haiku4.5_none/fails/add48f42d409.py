# -999
# error: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
import numpy as np

class Agent:
    """Demand-weighted allocation: send more to sinks with backlog or high forecast demand."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        layout = config["layout"]
        
        # Extract nominal capacities for each action slot
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        # Store routing and commodity info
        self.action_slots = action_slots
        self.edges = static["edges"]
        self.sinks = static["sinks"]
        self.commodities = static["commodities"]
        self.layout_demands = layout["demands"]  # list of [sink_node, commodity]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = np.zeros_like(self.capacity)
        action_mask = observation["action_mask"]
        
        # Compute urgency per sink (backlog + short-term demand forecast)
        demand_forecast = observation["demand_forecast.qty"]  # (num_demands, 8 weeks)
        backlog = observation["backlog.qty"]  # (num_demands,)
        demand_forecast_obs = observation["demand_forecast.qty.observed"]
        backlog_obs = observation["backlog.qty.observed"]
        
        num_demands = len(self.layout_demands)
        sink_urgency = np.zeros(num_demands)
        
        for sink_idx in range(num_demands):
            urgency = 0.0
            
            # High weight on backlog (unserved demand)
            if backlog_obs[sink_idx]:
                urgency += backlog[sink_idx] * 2.0
            
            # Weight near-term forecast (weeks 0-2)
            if demand_forecast_obs[sink_idx]:
                urgency += np.sum(demand_forecast[sink_idx, :3])
            
            sink_urgency[sink_idx] = urgency
        
        # Normalize urgency to [0, 1]
        max_urgency = np.max(sink_urgency) if np.max(sink_urgency) > 0 else 1.0
        sink_urgency = sink_urgency / max_urgency
        
        # Get freight costs to penalize expensive routes
        graph_c = observation["graph_now.c"]  # (num_edges,)
        graph_c_obs = observation["graph_now.c.observed"]
        
        # Send flows to routes leading to high-urgency sinks
        for slot_idx in range(len(flows)):
            if not action_mask[slot_idx]:
                continue
            
            edge_idx = self.action_slots["edge"][slot_idx]
            k_idx = self.action_slots["k"][slot_idx]
            dest_node = self.edges["head"][edge_idx]
            
            # Find which sinks this edge could serve
            # Routes lead eventually to sinks; use destination node as proxy
            max_sink_urgency = 0.0
            for sink_idx in range(num_demands):
                sink_node = self.layout_demands[sink_idx][0]
                sink_k = self.layout_demands[sink_idx][1]
                # Send on this route if it could serve this sink
                if sink_k == k_idx:
                    max_sink_urgency = max(max_sink_urgency, sink_urgency[sink_idx])
            
            # Base send: urgency-weighted capacity, minimum 50% to avoid starvation
            base_send = 0.5 + 0.5 * max_sink_urgency
            
            # Penalty for high freight cost
            if graph_c_obs[edge_idx]:
                cost_value = graph_c[edge_idx]
                mean_cost = np.mean(graph_c[graph_c_obs.astype(bool)])
                if cost_value > 1.5 * mean_cost:
                    base_send *= 0.85  # 15% penalty on expensive routes
            
            flows[slot_idx] = self.capacity[slot_idx] * base_send
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }