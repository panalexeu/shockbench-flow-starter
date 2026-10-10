# -0.025218561760539337
import numpy as np

class Agent:
    """Pipeline-aware: adjust allocation based on in-transit shipments."""
    
    def __init__(self, config=None):
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        self.action_slots = action_slots
        self.sinks = static["sinks"]
        self.commodities = static["commodities"]
        self.edges = static["edges"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = self.capacity.copy()
        action_mask = observation["action_mask"]
        
        # Estimate in-transit per commodity based on pipeline
        pipeline_k = observation["pipeline.k"]
        pipeline_qty = observation["pipeline.qty"]
        pipeline_obs = observation["pipeline.qty.observed"]
        
        num_commodities = len(self.commodities["id"])
        in_transit = np.zeros(num_commodities)
        
        if pipeline_obs.sum() > 0:
            for i in np.where(pipeline_obs)[0]:
                if i < len(pipeline_k):
                    k_idx = int(pipeline_k[i])
                    if k_idx < num_commodities:
                        in_transit[k_idx] += float(pipeline_qty[i])
        
        # Compute demand signals
        demand_forecast = observation["demand_forecast.qty"]
        backlog = observation["backlog.qty"]
        backlog_obs = observation["backlog.qty.observed"]
        
        urgency = np.zeros(num_commodities)
        near_term_demand = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            k_idx = self.sinks["k"][sink_idx]
            
            # Backlog is urgent
            if sink_idx < len(backlog_obs) and backlog_obs[sink_idx]:
                urgency[k_idx] += float(backlog[sink_idx]) * 3.0
            
            # Next 2 weeks demand
            if sink_idx < len(demand_forecast):
                near_term_demand[k_idx] += float(demand_forecast[sink_idx, 0])
                if len(demand_forecast[sink_idx]) > 1:
                    near_term_demand[k_idx] += float(demand_forecast[sink_idx, 1]) * 0.7
        
        # Compare demand against in-transit
        total_demand = np.sum(near_term_demand) + np.sum(urgency) + 1e-9
        total_in_transit = np.sum(in_transit)
        
        # If much in pipeline, reduce sends; if little, increase
        pipeline_pressure = total_in_transit / (total_demand + 1e-9)
        
        for slot_idx in range(len(flows)):
            if not action_mask[slot_idx]:
                flows[slot_idx] = 0
                continue
            
            k_idx = self.action_slots["k"][slot_idx]
            
            # If commodity is urgently needed, send more; otherwise, be conservative
            commodity_urgency = urgency[k_idx] + near_term_demand[k_idx]
            avg_urgency = total_demand / num_commodities
            
            # Adjustment: 0.8 base, up to 1.1 if urgent
            urgency_factor = 0.8 + 0.3 * min(commodity_urgency / (avg_urgency + 1e-9), 1.0)
            
            # Pipeline factor: reduce if too much in transit
            pipeline_factor = 1.0 if pipeline_pressure < 1.2 else (1.5 - pipeline_pressure)
            pipeline_factor = max(0.7, min(1.1, pipeline_factor))
            
            flows[slot_idx] *= (urgency_factor * pipeline_factor)
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }