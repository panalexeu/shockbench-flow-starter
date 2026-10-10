# -999
# error: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
import numpy as np

class Agent:
    """Dynamic allocation: backlog drives intensity, cost and threats adjust."""
    
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
        week = int(observation["week"][0])
        
        num_commodities = len(self.commodities["id"])
        
        # === COMPUTE COMMODITY PRESSURE: BACKLOG + FORECAST ===
        backlog = observation["backlog.qty"]
        backlog_obs = observation["backlog.qty.observed"]
        demand_forecast = observation["demand_forecast.qty"]
        demand_forecast_obs = observation["demand_forecast.qty.observed"]
        
        pressure = np.zeros(num_commodities)
        
        # Heavy weight on backlog (unserved demand)
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(backlog_obs) and backlog_obs[sink_idx]:
                k_idx = self.sinks["k"][sink_idx]
                pressure[k_idx] += backlog[sink_idx] * 1.5
        
        # Medium weight on near-term forecast
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(demand_forecast_obs) and demand_forecast_obs[sink_idx]:
                k_idx = self.sinks["k"][sink_idx]
                # Current + next week
                pressure[k_idx] += demand_forecast[sink_idx, 0] + demand_forecast[sink_idx, 1]
        
        # Normalize
        max_pressure = np.max(pressure) if np.max(pressure) > 0 else 1.0
        pressure_norm = pressure / max_pressure
        
        # === COST MODULATION ===
        graph_c = observation["graph_now.c"]
        graph_c_obs = observation["graph_now.c.observed"]
        graph_tariff = observation["graph_now.tariff"]
        graph_tariff_obs = observation["graph_now.tariff.observed"]
        
        mean_cost = 1.0
        if graph_c_obs.any():
            mean_cost = np.mean(graph_c[graph_c_obs.astype(bool)])
        
        cost_factor = np.ones(len(flows))
        for slot_idx in range(len(flows)):
            edge_idx = self.action_slots["edge"][slot_idx]
            k_idx = self.action_slots["k"][slot_idx]
            
            # Reduce on expensive routes
            if graph_c_obs[edge_idx]:
                if graph_c[edge_idx] > 1.3 * mean_cost:
                    cost_factor[slot_idx] *= 0.9
            
            # Reduce on high-tariff routes
            if graph_tariff_obs[edge_idx, k_idx]:
                tariff = graph_tariff[edge_idx, k_idx]
                if tariff > 0.10:
                    cost_factor[slot_idx] *= max(0.8, 1.0 - tariff * 0.5)
        
        # === THREAT BOOST ===
        pending_edge = observation["pending_prohibitions.edge"]
        pending_k = observation["pending_prohibitions.k"]
        pending_week = observation["pending_prohibitions.effective_week"]
        pending_obs = observation["pending_prohibitions.edge.observed"]
        
        threat_boost = np.ones(len(flows))
        if pending_obs.any():
            for i in np.where(pending_obs)[0]:
                edge_idx = int(pending_edge[i])
                k_idx = int(pending_k[i])
                eff_week = int(pending_week[i])
                weeks_until = max(0, eff_week - week)
                
                if 0 < weeks_until <= 2:
                    boost = 1.25
                elif 2 < weeks_until <= 4:
                    boost = 1.10
                else:
                    boost = 1.0
                
                for slot_idx in range(len(flows)):
                    if (self.action_slots["edge"][slot_idx] == edge_idx and
                        self.action_slots["k"][slot_idx] == k_idx):
                        threat_boost[slot_idx] = max(threat_boost[slot_idx], boost)
        
        # === FINAL ALLOCATION ===
        for slot_idx in range(len(flows)):
            if action_mask[slot_idx]:
                k_idx = self.action_slots["k"][slot_idx]
                # Base: 0.65 + 0.35 * pressure
                intensity = 0.65 + 0.35 * min(pressure_norm[k_idx], 1.0)
                flows[slot_idx] = self.capacity[slot_idx] * intensity * cost_factor[slot_idx] * threat_boost[slot_idx]
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }