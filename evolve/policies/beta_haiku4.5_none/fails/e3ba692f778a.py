# -999
# error: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
import numpy as np

class Agent:
    """Demand-responsive allocation with cost awareness and urgency weighting."""
    
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
        flows = np.zeros_like(self.capacity)
        action_mask = observation["action_mask"]
        
        num_commodities = len(self.commodities["id"])
        
        # === COMPUTE COMMODITY URGENCY ===
        backlog = observation["backlog.qty"]
        backlog_obs = observation["backlog.qty.observed"]
        demand_forecast = observation["demand_forecast.qty"]
        demand_forecast_obs = observation["demand_forecast.qty.observed"]
        
        urgency = np.zeros(num_commodities)
        
        # Aggregate backlog by commodity
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(backlog_obs) and backlog_obs[sink_idx]:
                k_idx = self.sinks["k"][sink_idx]
                urgency[k_idx] += backlog[sink_idx] * 2.0
        
        # Aggregate demand forecast (next 2 weeks) by commodity
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(demand_forecast_obs) and demand_forecast_obs[sink_idx]:
                k_idx = self.sinks["k"][sink_idx]
                # Weight: current week + next week heavily
                urgency[k_idx] += demand_forecast[sink_idx, 0] + 1.2 * demand_forecast[sink_idx, 1]
        
        # Normalize urgency to [0, 1]
        max_urgency = np.max(urgency) if np.max(urgency) > 0 else 1.0
        urgency_norm = urgency / max_urgency
        
        # === COMPUTE COST PENALTY ===
        graph_c = observation["graph_now.c"]
        graph_c_obs = observation["graph_now.c.observed"]
        graph_tariff = observation["graph_now.tariff"]
        graph_tariff_obs = observation["graph_now.tariff.observed"]
        
        mean_cost = 1.0
        if graph_c_obs.any():
            mean_cost = np.mean(graph_c[graph_c_obs.astype(bool)])
        
        cost_penalty = np.ones(len(flows))
        for slot_idx in range(len(flows)):
            edge_idx = self.action_slots["edge"][slot_idx]
            k_idx = self.action_slots["k"][slot_idx]
            
            # Penalty for high freight cost
            if graph_c_obs[edge_idx]:
                if graph_c[edge_idx] > 1.4 * mean_cost:
                    cost_penalty[slot_idx] *= 0.85
            
            # Penalty for high tariff
            if graph_tariff_obs[edge_idx, k_idx]:
                tariff = graph_tariff[edge_idx, k_idx]
                if tariff > 0.08:
                    cost_penalty[slot_idx] *= (1.0 - min(0.15, tariff))
        
        # === ALLOCATE FLOWS ===
        for slot_idx in range(len(flows)):
            if action_mask[slot_idx]:
                k_idx = self.action_slots["k"][slot_idx]
                # Boost: 0.7 baseline + 0.3 * urgency
                intensity = 0.7 + 0.3 * min(urgency_norm[k_idx], 1.0)
                flows[slot_idx] = self.capacity[slot_idx] * intensity * cost_penalty[slot_idx]
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }