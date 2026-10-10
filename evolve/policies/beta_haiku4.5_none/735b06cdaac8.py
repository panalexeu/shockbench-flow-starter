# 0.06815598438781861
import numpy as np

class Agent:
    """Conservative with gentle pipeline load balancing."""
    
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
        
        # Pipeline congestion data
        pipeline_edge = observation["pipeline.edge"]
        pipeline_k = observation["pipeline.k"]
        pipeline_qty = observation["pipeline.qty"]
        pipeline_observed = observation["pipeline.qty.observed"]
        
        # Build in-flight inventory by (edge, k)
        in_flight = {}
        if pipeline_observed.sum() > 0:
            for i in np.where(pipeline_observed)[0]:
                edge_idx = int(pipeline_edge[i])
                k_idx = int(pipeline_k[i])
                qty = float(pipeline_qty[i])
                key = (edge_idx, k_idx)
                in_flight[key] = in_flight.get(key, 0.0) + qty
        
        # Get backlog and demand to assess urgency
        backlog = observation["backlog.qty"]
        demand_forecast = observation["demand_forecast.qty"]
        last_served = observation["last_week.sinks.served"]
        last_demand = observation["last_week.sinks.demand"]
        
        shortage_rate = 1.0 - (np.sum(last_served) / (np.sum(last_demand) + 1e-6))
        backlog_total = np.sum(backlog)
        
        # Compute commodity urgency
        num_commodities = len(self.commodities["id"])
        urgency = np.zeros(num_commodities)
        for sink_idx in range(len(self.sinks["k"])):
            k = self.sinks["k"][sink_idx]
            if sink_idx < len(demand_forecast):
                urgency[k] += np.sum(demand_forecast[sink_idx, :2])
        
        total_urgency = np.sum(urgency) + 1e-6
        
        # Adjust flows based on pipeline load and urgency
        for slot_idx in range(len(flows)):
            if not action_mask[slot_idx]:
                flows[slot_idx] = 0.0
                continue
            
            edge_idx = self.action_slots["edge"][slot_idx]
            k_idx = self.action_slots["k"][slot_idx]
            key = (edge_idx, k_idx)
            
            cap = self.capacity[slot_idx]
            in_flight_qty = in_flight.get(key, 0.0)
            
            # Gentle pipeline throttle
            load_ratio = in_flight_qty / (cap + 1e-6) if cap > 0 else 0.0
            
            if load_ratio < 0.3:
                pipeline_factor = 1.0
            elif load_ratio < 0.7:
                # Gradual reduction 1.0 -> 0.9
                pipeline_factor = 1.0 - 0.1 * (load_ratio - 0.3) / 0.4
            elif load_ratio < 1.5:
                # Gradual reduction 0.9 -> 0.6
                pipeline_factor = 0.9 - 0.3 * (load_ratio - 0.7) / 0.8
            else:
                # Capped at 0.6 for very high loads
                pipeline_factor = 0.6
            
            # Urgency override: if this commodity is urgent, allow more despite congestion
            urgency_weight = urgency[k_idx] / total_urgency
            urgency_override = 1.0
            if urgency_weight > 0.1 and (shortage_rate > 0.2 or backlog_total > 10.0):
                urgency_override = 1.0 + min(0.2, urgency_weight * 0.5)
            
            allocation_factor = pipeline_factor * urgency_override
            flows[slot_idx] = cap * min(1.2, allocation_factor)
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }