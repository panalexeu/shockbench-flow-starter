# 0.3864771687092224
import numpy as np

class Agent:
    """Threat-aware with measured response: anticipate prohibitions, boost at risk routes."""
    
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
        week = int(observation["week"][0]) if len(observation["week"]) > 0 else 1
        
        # Check for imminent threats
        pending_edge = observation["pending_prohibitions.edge"]
        pending_k = observation["pending_prohibitions.k"]
        pending_week = observation["pending_prohibitions.effective_week"]
        pending_obs = observation["pending_prohibitions.edge.observed"]
        
        threat_boost = np.ones(len(flows))
        
        if pending_obs.sum() > 0:
            for i in np.where(pending_obs)[0]:
                edge_idx = int(pending_edge[i])
                k_idx = int(pending_k[i])
                eff_week = int(pending_week[i])
                weeks_until = max(0, eff_week - week)
                
                # Conservative boost: 1.0 to 1.2x based on timing
                if 0 < weeks_until <= 1:
                    boost = 1.2
                elif 1 < weeks_until <= 2:
                    boost = 1.12
                elif 2 < weeks_until <= 4:
                    boost = 1.06
                else:
                    boost = 1.0
                
                # Apply to matching slots
                for slot_idx in range(len(flows)):
                    slot_edge = self.action_slots["edge"][slot_idx]
                    slot_k = self.action_slots["k"][slot_idx]
                    if slot_edge == edge_idx and slot_k == k_idx:
                        threat_boost[slot_idx] = max(threat_boost[slot_idx], boost)
        
        flows = flows * threat_boost * action_mask
        
        # Add demand signal for base allocation
        demand_forecast = observation["demand_forecast.qty"]
        backlog = observation["backlog.qty"]
        backlog_obs = observation["backlog.qty.observed"]
        
        num_commodities = len(self.commodities["id"])
        commodity_urgency = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            k_idx = self.sinks["k"][sink_idx]
            if sink_idx < len(backlog_obs) and backlog_obs[sink_idx]:
                commodity_urgency[k_idx] += float(backlog[sink_idx]) * 2.0
            if sink_idx < len(demand_forecast):
                commodity_urgency[k_idx] += float(demand_forecast[sink_idx, 0]) * 0.8
        
        # Apply modest urgency weighting if no threat boost
        max_urg = np.max(commodity_urgency) if np.max(commodity_urgency) > 0 else 1.0
        for slot_idx in range(len(flows)):
            if action_mask[slot_idx] and threat_boost[slot_idx] == 1.0:
                k_idx = self.action_slots["k"][slot_idx]
                urg_norm = commodity_urgency[k_idx] / (max_urg + 1e-9)
                urgency_factor = 0.9 + 0.2 * min(urg_norm, 1.0)
                flows[slot_idx] *= urgency_factor
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }