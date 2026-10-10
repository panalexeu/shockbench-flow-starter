# 0.40086818017976117
import numpy as np

class Agent:
    """Aggressive backlog + forecast allocation with pipeline feedback."""
    
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
        
        num_commodities = len(self.commodities["id"])
        
        # Compute commodity-level pressure from backlog + forecast
        backlog = observation["backlog.qty"]
        backlog_per_k = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(backlog):
                k_idx = self.sinks["k"][sink_idx]
                backlog_per_k[k_idx] += backlog[sink_idx]
        
        demand_forecast = observation["demand_forecast.qty"]
        demand_per_k = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < demand_forecast.shape[0]:
                k_idx = self.sinks["k"][sink_idx]
                # Weight current and next 2 weeks heavily
                demand_per_k[k_idx] += demand_forecast[sink_idx, 0] + 0.9 * demand_forecast[sink_idx, 1] + 0.7 * demand_forecast[sink_idx, 2]
        
        # Pressure: 70% backlog, 30% forecast
        max_backlog = np.max(backlog_per_k) if np.max(backlog_per_k) > 0 else 1.0
        max_demand = np.max(demand_per_k) if np.max(demand_per_k) > 0 else 1.0
        
        backlog_norm = backlog_per_k / (max_backlog + 1e-9)
        demand_norm = demand_per_k / (max_demand + 1e-9)
        
        pressure = 0.7 * backlog_norm + 0.3 * demand_norm
        
        # Pipeline feedback: if pipeline low, boost; if high, reduce
        pipeline_k = observation["pipeline.k"]
        pipeline_qty = observation["pipeline.qty"]
        pipeline_obs = observation["pipeline.qty.observed"]
        
        pipeline_per_k = np.zeros(num_commodities)
        for i in range(len(pipeline_k)):
            if i < len(pipeline_obs) and pipeline_obs[i]:
                k_idx = int(pipeline_k[i])
                if k_idx < num_commodities:
                    pipeline_per_k[k_idx] += pipeline_qty[i]
        
        max_pipeline = np.max(pipeline_per_k) if np.max(pipeline_per_k) > 0 else 1.0
        pipeline_norm = pipeline_per_k / (max_pipeline + 1e-9)
        
        # Stock level feedback
        stock = observation["stock.qty"]
        stock_obs = observation["stock.qty.observed"]
        total_stock = np.sum(stock[stock_obs == 1]) if stock_obs.sum() > 0 else 0.0
        
        # Apply allocation
        for slot_idx in range(len(flows)):
            if action_mask[slot_idx]:
                k_idx = self.action_slots["k"][slot_idx]
                
                # Base intensity: 0.80 + 0.20 * pressure
                intensity = 0.80 + 0.20 * min(pressure[k_idx], 1.0)
                
                # Pipeline feedback: boost if low, reduce if high
                pipeline_feedback = 1.0 + 0.15 * (1.0 - pipeline_norm[k_idx])
                
                # Stock feedback: if total stock very low, urgency multiplier
                if total_stock < 10.0:
                    stock_feedback = 1.10
                else:
                    stock_feedback = 1.0
                
                flows[slot_idx] *= intensity * pipeline_feedback * stock_feedback
            else:
                flows[slot_idx] = 0.0
        
        # Soft cap
        allowed_total = np.sum(self.capacity * action_mask)
        total_flows = np.sum(flows)
        if total_flows > allowed_total * 1.3:
            flows *= (allowed_total * 1.25 / (total_flows + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }