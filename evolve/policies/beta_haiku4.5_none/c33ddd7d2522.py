# 0.0963782808991654
import numpy as np

class Agent:
    """Threat-aware pre-staging: boost flow on routes facing imminent prohibitions."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
        
        self.action_slots = action_slots
        self.sinks = static["sinks"]
        self.commodities = static["commodities"]
    
    def act(self, observation):
        flows = self.capacity * observation["action_mask"]
        action_mask = observation["action_mask"]
        current_week = int(observation["week"][0])
        
        # Identify threatened routes from pending prohibitions
        pending_edge = observation["pending_prohibitions.edge"]
        pending_k = observation["pending_prohibitions.k"]
        pending_week = observation["pending_prohibitions.effective_week"]
        pending_obs = observation["pending_prohibitions.edge.observed"]
        
        threat_boost = np.ones(len(flows))
        
        num_pending = int(pending_obs.sum())
        if num_pending > 0:
            for i in range(len(pending_edge)):
                if not pending_obs[i]:
                    continue
                edge_idx = int(pending_edge[i])
                k_idx = int(pending_k[i])
                eff_week = int(pending_week[i])
                weeks_until = eff_week - current_week
                
                if 0 < weeks_until <= 3:
                    boost_amt = 1.3 if weeks_until <= 1 else 1.2 if weeks_until <= 2 else 1.1
                    for slot_idx in range(len(flows)):
                        if (self.action_slots["edge"][slot_idx] == edge_idx and
                            self.action_slots["k"][slot_idx] == k_idx):
                            threat_boost[slot_idx] = max(threat_boost[slot_idx], boost_amt)
        
        # Base intensity from backlog
        backlog = observation["backlog.qty"]
        backlog_obs = observation["backlog.qty.observed"]
        num_commodities = len(self.commodities["id"])
        
        backlog_per_k = np.zeros(num_commodities)
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(backlog_obs) and backlog_obs[sink_idx] > 0:
                k_idx = self.sinks["k"][sink_idx]
                backlog_per_k[k_idx] += backlog[sink_idx]
        
        max_backlog = np.max(backlog_per_k) if np.max(backlog_per_k) > 0 else 1.0
        backlog_norm = backlog_per_k / (max_backlog + 1e-9)
        
        # Apply modulation
        for slot_idx in range(len(flows)):
            if action_mask[slot_idx]:
                k_idx = self.action_slots["k"][slot_idx]
                base_intensity = 0.7 + 0.3 * backlog_norm[k_idx]
                flows[slot_idx] *= base_intensity * threat_boost[slot_idx]
        
        # Cap to capacity constraints
        allowed_total = np.sum(self.capacity * action_mask)
        flows_total = np.sum(flows)
        if flows_total > allowed_total * 1.15:
            flows = flows * (allowed_total * 1.1 / (flows_total + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }