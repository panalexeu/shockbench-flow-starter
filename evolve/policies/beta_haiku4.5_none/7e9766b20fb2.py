# 0.22363324612509672
import numpy as np

class Agent:
    """Conservative with disruption-aware pre-buffering."""
    
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
        
        # Disruption signals
        warning_score = observation["warning.score"]
        graph_open = observation["graph_now.open"]
        
        max_warning = np.max(warning_score) if warning_score.size > 0 else 0.0
        min_open = np.min(graph_open) if graph_open.size > 0 else 1.0
        
        disruption_level = max(max_warning * 0.2, (1.0 - min_open) * 0.3)
        
        # Backlog and demand
        backlog = observation["backlog.qty"]
        demand_forecast = observation["demand_forecast.qty"]
        num_commodities = len(self.commodities["id"])
        
        urgency_per_k = np.zeros(num_commodities)
        for sink_idx in range(len(self.sinks["k"])):
            k_idx = self.sinks["k"][sink_idx]
            if sink_idx < len(backlog):
                urgency_per_k[k_idx] += backlog[sink_idx]
            if sink_idx < demand_forecast.shape[0]:
                urgency_per_k[k_idx] += demand_forecast[sink_idx, 0]
        
        max_urgency = np.max(urgency_per_k) if np.max(urgency_per_k) > 1e-6 else 1.0
        urgency_norm = urgency_per_k / (max_urgency + 1e-9)
        
        # Lead time
        edge_tau = np.array(self.edges["tau0"], dtype=float)
        
        # Allocation: base + urgency + disruption pre-buffer + lead-time priority
        for slot_idx in range(len(flows)):
            k_idx = self.action_slots["k"][slot_idx]
            edge_idx = self.action_slots["edge"][slot_idx]
            
            # Base: 0.70
            base = 0.70
            # Urgency boost: up to 0.25
            urgency_boost = 0.25 * urgency_norm[k_idx]
            # Disruption pre-buffer: scale with disruption level
            disruption_boost = 0.15 * disruption_level
            # Lead-time priority: longer routes get modest boost
            lead_boost = 0.05 * min(edge_tau[edge_idx] / 6.0, 1.0)
            
            allocation_factor = base + urgency_boost + disruption_boost + lead_boost
            flows[slot_idx] *= allocation_factor
        
        # Soft cap
        allowed_capacity = self.capacity * action_mask
        total_allowed = np.sum(allowed_capacity)
        total_flows = np.sum(flows)
        if total_flows > total_allowed * 1.2:
            flows = flows * (total_allowed * 1.15 / (total_flows + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }