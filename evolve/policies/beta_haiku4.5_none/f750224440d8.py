# 0.4699062375097181
import numpy as np

class Agent:
    """Combined: backlog + threat + short lead-time, with moderate boosting."""
    
    def __init__(self, config=None):
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        u0 = static["edges"]["u0"]
        tau0 = static["edges"]["tau0"]
        action_slots = static["action_slots"]
        
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        self.edge_tau = np.array(tau0, dtype=float)
        
        self.action_slots = action_slots
        self.sinks = static["sinks"]
        self.commodities = static["commodities"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = self.capacity.copy()
        action_mask = observation["action_mask"]
        week = int(observation["week"][0])
        
        num_commodities = len(self.commodities["id"])
        
        # === Backlog signal ===
        backlog = observation["backlog.qty"]
        backlog_per_k = np.zeros(num_commodities)
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(backlog):
                k_idx = self.sinks["k"][sink_idx]
                backlog_per_k[k_idx] += backlog[sink_idx]
        
        # === Threat signal ===
        pending_edge = observation["pending_prohibitions.edge"]
        pending_k = observation["pending_prohibitions.k"]
        pending_week = observation["pending_prohibitions.effective_week"]
        pending_obs = observation["pending_prohibitions.edge.observed"]
        
        threat_map = {}
        for i in range(len(pending_obs)):
            if pending_obs[i] == 1:
                edge_idx = int(pending_edge[i])
                k_idx = int(pending_k[i])
                eff_week = int(pending_week[i])
                weeks_until = max(0, eff_week - week)
                
                boost = 1.0
                if 0 < weeks_until <= 1:
                    boost = 1.4
                elif 1 < weeks_until <= 2:
                    boost = 1.25
                elif 2 < weeks_until <= 4:
                    boost = 1.1
                
                key = (edge_idx, k_idx)
                threat_map[key] = max(threat_map.get(key, 1.0), boost)
        
        # === Apply factors per slot ===
        for slot_idx in range(len(flows)):
            edge_idx = self.action_slots["edge"][slot_idx]
            k_idx = self.action_slots["k"][slot_idx]
            
            # Backlog boost: 1.2x if this commodity has backlog
            backlog_boost = 1.2 if backlog_per_k[k_idx] > 0 else 1.0
            
            # Threat boost
            threat_boost = threat_map.get((edge_idx, k_idx), 1.0)
            
            # Lead-time boost: prefer short routes
            tau = self.edge_tau[edge_idx]
            if tau <= 1:
                lead_boost = 1.15
            elif tau <= 2:
                lead_boost = 1.08
            else:
                lead_boost = 1.0
            
            flows[slot_idx] *= backlog_boost * threat_boost * lead_boost
        
        flows = flows * action_mask
        
        # Soft cap
        allowed_total = np.sum(self.capacity * action_mask)
        flows_total = np.sum(flows)
        if flows_total > allowed_total * 1.2:
            flows = flows * (allowed_total * 1.15 / (flows_total + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }