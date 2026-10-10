# 0.4699062375097181
import numpy as np

class Agent:
    """Aggressive backlog-driven with lead-time frontloading."""
    
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
        self.edges = static["edges"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = self.capacity.copy()
        action_mask = observation["action_mask"]
        flows = flows * action_mask
        
        # Compute backlog and pipeline per commodity
        backlog = observation["backlog.qty"]
        pipeline_qty = observation["pipeline.qty"]
        pipeline_k = observation["pipeline.k"]
        pipeline_obs = observation["pipeline.qty.observed"]
        
        num_commodities = len(self.commodities["id"])
        backlog_per_k = np.zeros(num_commodities)
        pipeline_per_k = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(backlog):
                k_idx = self.sinks["k"][sink_idx]
                backlog_per_k[k_idx] += backlog[sink_idx]
        
        for i in np.where(pipeline_obs)[0]:
            k_idx = int(pipeline_k[i])
            if k_idx < num_commodities:
                pipeline_per_k[k_idx] += pipeline_qty[i]
        
        # Strong backlog pressure
        max_backlog = np.max(backlog_per_k) if np.max(backlog_per_k) > 0 else 1.0
        backlog_pressure = backlog_per_k / (max_backlog + 1e-9)
        
        # Lead-time boost: apply only when both backlog AND low pipeline
        max_pipeline = np.max(pipeline_per_k) if np.max(pipeline_per_k) > 0 else 1.0
        pipeline_depletion = 1.0 - (pipeline_per_k / (max_pipeline + 1e-9))
        
        # Only apply lead-time boost when both signals are high
        lead_time_condition = (backlog_pressure > 0.3) * (pipeline_depletion > 0.3)
        edge_tau = np.array(self.edges["tau0"], dtype=float)
        
        # Apply aggressive boost: up to 1.30x on backlog, up to 1.10x on lead time
        for slot_idx in range(len(flows)):
            k_idx = self.action_slots["k"][slot_idx]
            edge_idx = self.action_slots["edge"][slot_idx]
            
            backlog_boost = 1.0 + 0.30 * backlog_pressure[k_idx]
            
            lead_time_boost = 1.0
            if lead_time_condition[k_idx] and edge_tau[edge_idx] > 2:
                tau_factor = min((edge_tau[edge_idx] - 1.0) / 5.0, 1.0)  # Scale by lead time, cap at 1.0
                lead_time_boost = 1.0 + 0.10 * tau_factor
            
            flows[slot_idx] *= (backlog_boost * lead_time_boost)
        
        # Hard cap to respect total capacity
        allowed_capacity = self.capacity * action_mask
        total_allowed = np.sum(allowed_capacity)
        total_flows = np.sum(flows)
        if total_flows > total_allowed:
            flows = flows * (total_allowed / (total_flows + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }