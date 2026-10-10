# 0.47027900374338105
import numpy as np

class Agent:
    """Enhanced backlog-driven with pipeline depletion signal."""
    
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
        
        # Compute boost: high backlog and low pipeline both trigger boost
        max_backlog = np.max(backlog_per_k) if np.max(backlog_per_k) > 0 else 1.0
        max_pipeline = np.max(pipeline_per_k) if np.max(pipeline_per_k) > 0 else 1.0
        
        backlog_pressure = backlog_per_k / (max_backlog + 1e-9)
        pipeline_pressure = 1.0 - (pipeline_per_k / (max_pipeline + 1e-9))  # Inverted: low pipeline = high pressure
        
        # Combined pressure: weight backlog more heavily
        combined_pressure = 0.7 * backlog_pressure + 0.3 * pipeline_pressure
        
        # Apply boost: up to 1.20x
        for slot_idx in range(len(flows)):
            k_idx = self.action_slots["k"][slot_idx]
            boost_factor = 1.0 + 0.20 * combined_pressure[k_idx]
            flows[slot_idx] *= boost_factor
        
        # Soft cap to avoid exceeding reasonable bounds
        allowed_capacity = self.capacity * action_mask
        total_allowed = np.sum(allowed_capacity)
        total_flows = np.sum(flows)
        if total_flows > total_allowed * 1.25:
            flows = flows * (total_allowed * 1.2 / (total_flows + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }