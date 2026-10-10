# 0.4699062375097181
import numpy as np

class Agent:
    """Pipeline-timing aware: frontload routes with delayed arrivals, reduce when pipeline full."""
    
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
        
        current_week = int(observation["week"][0])
        
        # Compute backlog per commodity
        backlog = observation["backlog.qty"]
        backlog_obs = observation["backlog.qty.observed"]
        num_commodities = len(self.commodities["id"])
        backlog_per_k = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(backlog_obs) and backlog_obs[sink_idx]:
                k_idx = self.sinks["k"][sink_idx]
                backlog_per_k[k_idx] += backlog[sink_idx]
        
        # Pipeline per commodity, segmented by arrival timing
        pipeline_qty = observation["pipeline.qty"]
        pipeline_k = observation["pipeline.k"]
        pipeline_arrival = observation["pipeline.arrival_week"]
        pipeline_obs = observation["pipeline.qty.observed"]
        
        pipeline_near = np.zeros(num_commodities)  # Arriving current + next week
        pipeline_far = np.zeros(num_commodities)   # Arriving 2-4 weeks out
        
        if pipeline_obs.any():
            for i in np.where(pipeline_obs)[0]:
                k_idx = int(pipeline_k[i])
                arrival = int(pipeline_arrival[i])
                if k_idx < num_commodities:
                    if arrival <= current_week + 1:
                        pipeline_near[k_idx] += pipeline_qty[i]
                    elif arrival <= current_week + 4:
                        pipeline_far[k_idx] += pipeline_qty[i]
        
        # Compute urgency: backlog minus near-term pipeline
        net_urgency = np.maximum(0, backlog_per_k - pipeline_near)
        max_urgency = np.max(net_urgency) if np.max(net_urgency) > 0 else 1.0
        urgency_norm = net_urgency / (max_urgency + 1e-9)
        
        # Compute pipeline depletion risk: when is far pipeline running dry?
        max_far_pipeline = np.max(pipeline_far) if np.max(pipeline_far) > 0 else 1.0
        pipeline_depletion = 1.0 - (pipeline_far / (max_far_pipeline + 1e-9))
        
        # Apply boosts per slot
        for slot_idx in range(len(flows)):
            k_idx = self.action_slots["k"][slot_idx]
            edge_idx = self.action_slots["edge"][slot_idx]
            
            # Backlog boost (always on)
            backlog_boost = 1.0 + 0.35 * urgency_norm[k_idx]
            
            # Pipeline depletion boost: when far-pipeline is low AND lead time is high
            pipeline_depletion_boost = 1.0
            if pipeline_depletion[k_idx] > 0.4 and edge_idx < len(self.edges["tau0"]):
                tau = self.edges["tau0"][edge_idx]
                if tau >= 3:  # Only boost long routes
                    tau_factor = min((tau - 2.0) / 4.0, 1.0)
                    pipeline_depletion_boost = 1.0 + 0.15 * tau_factor * pipeline_depletion[k_idx]
            
            flows[slot_idx] *= (backlog_boost * pipeline_depletion_boost)
        
        # Hard cap
        allowed_capacity = self.capacity * action_mask
        total_allowed = np.sum(allowed_capacity)
        total_flows = np.sum(flows)
        if total_flows > total_allowed * 1.25:
            flows = flows * (total_allowed * 1.25 / (total_flows + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }