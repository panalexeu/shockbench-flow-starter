# -999
# error: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
import numpy as np

class Agent:
    """Policy 3: Balanced demand + disruption integration with lead-time awareness."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        layout = config["layout"]
        
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        self.action_slots = action_slots
        self.edges = static["edges"]
        self.sinks = static["sinks"]
        self.layout_demands = layout["demands"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = np.zeros_like(self.capacity)
        action_mask = observation["action_mask"]
        week = int(observation["week"][0])
        
        # === DEMAND ASSESSMENT ===
        demand_forecast = observation["demand_forecast.qty"]
        backlog = observation["backlog.qty"]
        demand_forecast_obs = observation["demand_forecast.qty.observed"]
        backlog_obs = observation["backlog.qty.observed"]
        
        num_commodities = len(self.sinks["k"])
        commodity_demand = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.layout_demands)):
            k = self.sinks["k"][sink_idx]
            if backlog_obs[sink_idx]:
                commodity_demand[k] += backlog[sink_idx] * 2.0
            if demand_forecast_obs[sink_idx]:
                commodity_demand[k] += np.sum(demand_forecast[sink_idx, :3])
        
        max_demand = np.max(commodity_demand) if np.max(commodity_demand) > 0 else 1.0
        commodity_demand = commodity_demand / (max_demand + 1e-9)
        
        # === DISRUPTION ASSESSMENT ===
        # Pending prohibitions: boost threatened (edge, k)
        pending_edge = observation["pending_prohibitions.edge"]
        pending_k = observation["pending_prohibitions.k"]
        pending_week = observation["pending_prohibitions.effective_week"]
        pending_obs = observation["pending_prohibitions.edge.observed"]
        
        threatened = {}  # (edge, k) -> boost
        max_threat = 0.0
        if pending_obs.any():
            for i in np.where(pending_obs)[0]:
                edge_idx = int(pending_edge[i])
                k_idx = int(pending_k[i])
                eff_week = int(pending_week[i])
                weeks_until = max(0, eff_week - week)
                
                if 0 < weeks_until <= 2:
                    boost = 1.5
                elif 2 < weeks_until <= 4:
                    boost = 1.2
                else:
                    boost = 1.0
                threatened[(edge_idx, k_idx)] = max(threatened.get((edge_idx, k_idx), 1.0), boost)
                max_threat = max(max_threat, weeks_until > 0 and weeks_until <= 4)
        
        # Warning score: elevated risk on certain routes/chokepoints
        warning_score = observation["warning.score"]
        max_warning = np.max(warning_score) if warning_score.size > 0 else 0.0
        disruption_intensity = max_warning * 0.3 + (1.0 if max_threat else 0.0) * 0.3
        
        # === FLOW ALLOCATION ===
        edge_tau = np.array(self.edges["tau0"], dtype=float)
        
        for slot_idx in range(len(flows)):
            if not action_mask[slot_idx]:
                continue
            
            edge_idx = self.action_slots["edge"][slot_idx]
            k = self.action_slots["k"][slot_idx]
            
            # Demand factor: 0.5 to 1.0 scaled by commodity demand
            demand_factor = 0.5 + 0.5 * commodity_demand[k]
            
            # Threat boost: if route threatened, amplify
            threat_boost = threatened.get((edge_idx, k), 1.0)
            
            # Lead-time boost: long routes get priority when disruption risk is high
            lead_time = edge_tau[edge_idx] if edge_tau[edge_idx] > 0 else 1.0
            lead_boost = 1.0 + 0.2 * min(lead_time / 6.0, 1.0) * disruption_intensity
            
            combined_factor = demand_factor * threat_boost * lead_boost
            # Cap combined factor to avoid excessive overshoot
            combined_factor = min(combined_factor, 1.5)
            
            flows[slot_idx] = self.capacity[slot_idx] * combined_factor
        
        # Clip to allowed capacity
        allowed = self.capacity * action_mask
        flows = np.minimum(flows, allowed)
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }