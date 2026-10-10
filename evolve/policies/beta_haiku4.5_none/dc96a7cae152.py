# 0.4565345237935042
import numpy as np

class Agent:
    """Risk-aware: boost flow on safe routes when disruption warnings are elevated."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        self.action_slots = action_slots
        self.commodities = static["commodities"]
        self.sinks = static["sinks"]
        self.edges = static["edges"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = np.zeros_like(self.capacity)
        action_mask = observation["action_mask"]
        
        # Detect disruption risk
        warning_score = observation["warning.score"]
        max_warning = np.max(warning_score) if warning_score.size > 0 else 0.0
        
        graph_open = observation["graph_now.open"]  # Chokepoint openness
        min_open = np.min(graph_open) if graph_open.size > 0 else 1.0
        
        # Compute demand urgency
        demand_forecast = observation["demand_forecast.qty"]
        backlog = observation["backlog.qty"]
        
        num_commodities = len(self.commodities["id"])
        urgency = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            k_idx = self.sinks["k"][sink_idx]
            if sink_idx < len(demand_forecast):
                urgency[k_idx] += demand_forecast[sink_idx, 0] + demand_forecast[sink_idx, 1] * 0.9
            if sink_idx < len(backlog):
                urgency[k_idx] += backlog[sink_idx] * 2.0
        
        total_urgency = np.sum(urgency) + 1e-6
        
        # Risk multiplier: higher when warnings or chokepoint closures detected
        risk_level = max(max_warning * 0.3, (1.0 - min_open) * 0.5)
        base_factor = 0.9 + 0.25 * min(risk_level, 1.0)  # 0.9 to 1.15
        
        # Allocate all allowed slots with modest urgency weighting and risk boost
        for slot_idx in range(len(flows)):
            if action_mask[slot_idx]:
                k_idx = self.action_slots["k"][slot_idx]
                urgency_weight = urgency[k_idx] / total_urgency
                # Modest urgency boost: 0.9x to 1.0x
                urgency_factor = 0.95 + 0.05 * min(urgency_weight * len(self.commodities["id"]), 1.0)
                flows[slot_idx] = self.capacity[slot_idx] * base_factor * urgency_factor
        
        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}