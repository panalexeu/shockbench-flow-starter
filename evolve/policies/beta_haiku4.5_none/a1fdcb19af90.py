# 0.4699062375097181
import numpy as np

class Agent:
    """Aggressive backlog boost + early disruption awareness: extend 0.47 with threat detection."""
    
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
        week = int(observation["week"][0])
        
        # Backlog per commodity
        backlog = observation["backlog.qty"]
        num_commodities = len(self.commodities["id"])
        backlog_per_commodity = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(backlog):
                k_idx = self.sinks["k"][sink_idx]
                backlog_per_commodity[k_idx] += backlog[sink_idx]
        
        # Normalize backlog
        max_backlog = np.max(backlog_per_commodity)
        if max_backlog > 1e-6:
            backlog_pressure = backlog_per_commodity / (max_backlog + 1e-9)
        else:
            backlog_pressure = np.zeros(num_commodities)
        
        # Detect upcoming prohibitions (aggressive window: 0-3 weeks)
        threat_boost = np.ones(len(flows))
        pending_edge = observation["pending_prohibitions.edge"]
        pending_k = observation["pending_prohibitions.k"]
        pending_week = observation["pending_prohibitions.effective_week"]
        pending_obs = observation["pending_prohibitions.edge.observed"]
        
        if pending_obs.any():
            for i in np.where(pending_obs)[0]:
                edge_idx = int(pending_edge[i])
                k_idx = int(pending_k[i])
                eff_week = int(pending_week[i])
                weeks_until = max(0, eff_week - week)
                
                # Boost if imminent (0-3 weeks)
                if 0 < weeks_until <= 1:
                    boost = 1.5
                elif 1 < weeks_until <= 2:
                    boost = 1.3
                elif 2 < weeks_until <= 3:
                    boost = 1.1
                else:
                    boost = 1.0
                
                # Apply to matching slots
                for slot_idx in range(len(flows)):
                    if (self.action_slots["edge"][slot_idx] == edge_idx and
                        self.action_slots["k"][slot_idx] == k_idx):
                        threat_boost[slot_idx] = max(threat_boost[slot_idx], boost)
        
        # Apply boosts: backlog + threat
        flows = flows * action_mask
        for slot_idx in range(len(flows)):
            k_idx = self.action_slots["k"][slot_idx]
            # Backlog: 1.0 to 1.20x
            backlog_boost = 1.0 + 0.20 * backlog_pressure[k_idx]
            flows[slot_idx] *= backlog_boost * threat_boost[slot_idx]
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }