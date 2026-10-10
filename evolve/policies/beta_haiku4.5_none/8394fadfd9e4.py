# -0.0434260988810418
import numpy as np

class Agent:
    """Lead-time aware: buffer inventory on routes with long delays before disruptions."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        self.action_slots = action_slots
        self.edges = static["edges"]
        self.sinks = static["sinks"]
        self.commodities = static["commodities"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = np.zeros_like(self.capacity)
        action_mask = observation["action_mask"]
        week = int(observation["week"][0])
        
        # Get lead times and demand
        graph_tau = observation["graph_now.tau"]
        demand_forecast = observation["demand_forecast.qty"]
        backlog = observation["backlog.qty"]
        last_served = observation["last_week.sinks.served"]
        last_demand = observation["last_week.sinks.demand"]
        
        # Check for announced threats
        pending_prohibitions_edge = observation["pending_prohibitions.edge"]
        pending_prohibitions_k = observation["pending_prohibitions.k"]
        pending_prohibitions_week = observation["pending_prohibitions.effective_week"]
        pending_observed = observation["pending_prohibitions.edge.observed"]
        
        # Build threat map: (edge, k) -> weeks_until_threat
        threat_map = {}
        if pending_observed.sum() > 0:
            for i in np.where(pending_observed)[0]:
                edge_idx = int(pending_prohibitions_edge[i])
                k_idx = int(pending_prohibitions_k[i])
                eff_week = int(pending_prohibitions_week[i])
                weeks_until = max(0, eff_week - week)
                key = (edge_idx, k_idx)
                threat_map[key] = min(threat_map.get(key, 999), weeks_until)
        
        # Compute commodity urgency
        num_commodities = len(self.commodities["id"])
        urgency = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            k = self.sinks["k"][sink_idx]
            if sink_idx < len(backlog):
                urgency[k] += backlog[sink_idx] * 3.0
            if sink_idx < len(demand_forecast):
                urgency[k] += np.sum(demand_forecast[sink_idx, :3]) * 0.8
        
        shortage_rate = 1.0 - (np.sum(last_served) / (np.sum(last_demand) + 1e-6))
        
        # Allocate
        total_urgency = np.sum(urgency) + 1e-6
        
        for slot_idx in range(len(flows)):
            if not action_mask[slot_idx]:
                continue
            
            edge_idx = self.action_slots["edge"][slot_idx]
            k_idx = self.action_slots["k"][slot_idx]
            
            # Base urgency-driven allocation
            urgency_weight = urgency[k_idx] / total_urgency
            base_factor = 0.5 + 0.5 * min(urgency_weight * 8.0, 1.0)
            
            # Lead-time boost: on long routes, send more to buffer
            lead_time_boost = 1.0
            if edge_idx < len(graph_tau):
                tau = int(graph_tau[edge_idx])
                if tau > 3:
                    lead_time_boost = 1.0 + 0.15 * min((tau - 3) / 3.0, 1.0)
            
            # Threat boost: send more on routes about to be disrupted
            threat_boost = 1.0
            threat_key = (edge_idx, k_idx)
            if threat_key in threat_map:
                weeks_until = threat_map[threat_key]
                if weeks_until <= 3:
                    # Heavily boost shipments in final weeks before closure
                    threat_boost = 1.0 + (3.0 - weeks_until) * 0.25
            
            # Shortage boost
            shortage_boost = 1.0 + shortage_rate * 0.25
            
            allocation_factor = base_factor * lead_time_boost * threat_boost * shortage_boost
            flows[slot_idx] = self.capacity[slot_idx] * min(1.3, allocation_factor)
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }