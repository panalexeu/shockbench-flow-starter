# 0.28523454873877835
import numpy as np

class Agent:
    """Advanced: balance cost, shortages, and disruption signals."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
        
        self.edges = static["edges"]
        self.sinks = static["sinks"]
        self.action_slots = action_slots
        self.commodities = static["commodities"]
        self.rng = np.random.default_rng(config["policy_seed"])
    
    def act(self, observation):
        flows = np.zeros_like(self.capacity)
        week = int(observation["week"][0])
        
        # Extract key state
        demand_forecast = observation["demand_forecast.qty"]  # (num_demands, 8)
        demand_observed = observation["demand_forecast.qty.observed"]
        stock = observation["stock.qty"]
        backlog = observation["backlog.qty"]
        action_mask = observation["action_mask"]
        
        # Get cost and shortage info
        last_cost = observation["last_week.cost_components"]  # indexed by cost_components
        last_served = observation["last_week.sinks.served"]
        last_demand = observation["last_week.sinks.demand"]
        
        # Identify high-cost or high-shortage weeks
        total_cost = np.sum(last_cost)
        shortage_rate = 1.0 - (np.sum(last_served) / (np.sum(last_demand) + 1e-6))
        
        # Get current warning signals for disruptions
        warning_score = observation["warning.score"]  # shape (num_warning_units,)
        warning_observed = observation["warning.score.observed"]
        
        # Check for announced disruptions
        pending_prohibitions_k = observation["pending_prohibitions.k"]
        pending_prohibitions_edge = observation["pending_prohibitions.edge"]
        pending_prohibitions_week = observation["pending_prohibitions.effective_week"]
        pending_obs = observation["pending_prohibitions.edge.observed"]
        
        # Compute commodity criticality based on shortage and value
        commodity_criticality = np.zeros(len(self.commodities["id"]))
        if demand_observed.sum() > 0:
            for i, (demand_avg, served, demanded) in enumerate(zip(demand_forecast[:, 0], last_served, last_demand)):
                k = self.sinks["k"][i]
                if demanded > 0:
                    shortage = 1.0 - (served / demanded)
                    value = self.commodities["v"][k]
                    commodity_criticality[k] += (shortage + 0.5) * value
        
        # Allocation strategy
        for slot_idx in range(len(flows)):
            if not action_mask[slot_idx]:
                continue
            
            edge_idx = self.action_slots["edge"][slot_idx]
            k = self.action_slots["k"][slot_idx]
            
            # Check if this edge is under threat from pending prohibitions
            threat_factor = 1.0
            if pending_obs.sum() > 0:
                for i in np.where(pending_obs)[0]:
                    if pending_prohibitions_edge[i] == edge_idx and pending_prohibitions_k[i] == k:
                        effective_week = int(pending_prohibitions_week[i])
                        weeks_until = max(0, effective_week - week)
                        if weeks_until <= 4:  # Threat within 4 weeks
                            threat_factor = min(1.0, 1.0 + (4 - weeks_until) * 0.2)
            
            # Send more on threatened routes (build inventory before closure)
            criticality_boost = 1.0 + commodity_criticality[k] / (np.sum(commodity_criticality) + 1.0)
            
            # Adaptive allocation: use more capacity when high shortage or high threat
            allocation_factor = min(1.0, 0.5 + 0.5 * shortage_rate) * threat_factor * min(1.5, criticality_boost)
            flows[slot_idx] = self.capacity[slot_idx] * allocation_factor
        
        # Normalize to available capacity if total exceeds reasonable bounds
        mask_capacity = self.capacity * action_mask
        if flows.sum() > mask_capacity.sum():
            flows = flows * (mask_capacity.sum() / (flows.sum() + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }