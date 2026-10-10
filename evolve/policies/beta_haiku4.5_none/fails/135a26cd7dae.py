# -999
# error: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
import numpy as np

class Agent:
    """Backlog-driven with demand-weighting and tariff-aware penalty."""
    
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
        flows = flows * action_mask
        
        # Compute backlog per commodity
        backlog = observation["backlog.qty"]
        num_commodities = len(self.commodities["id"])
        backlog_per_k = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(backlog):
                k_idx = self.sinks["k"][sink_idx]
                backlog_per_k[k_idx] += backlog[sink_idx]
        
        # Backlog pressure
        max_backlog = np.max(backlog_per_k) if np.max(backlog_per_k) > 0 else 1.0
        backlog_pressure = backlog_per_k / (max_backlog + 1e-9)
        
        # Compute demand pressure from forecast
        demand_forecast = observation["demand_forecast.qty"]
        demand_forecast_obs = observation["demand_forecast.qty.observed"]
        demand_per_k = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(demand_forecast) and demand_forecast_obs[sink_idx]:
                k_idx = self.sinks["k"][sink_idx]
                # Weight near-term forecast heavily
                demand_per_k[k_idx] += 2.0 * demand_forecast[sink_idx, 0] + np.sum(demand_forecast[sink_idx, 1:3])
        
        max_demand = np.max(demand_per_k) if np.max(demand_per_k) > 0 else 1.0
        demand_pressure = demand_per_k / (max_demand + 1e-9)
        
        # Combined: backlog and near-term demand
        combined_pressure = 0.6 * backlog_pressure + 0.4 * demand_pressure
        
        # Get tariff info
        tariff = observation["graph_now.tariff"]
        tariff_obs = observation["graph_now.tariff.observed"]
        
        # Compute mean tariff for penalty scaling
        if tariff_obs.sum() > 0:
            mean_tariff = np.mean(tariff[tariff_obs.astype(bool)])
        else:
            mean_tariff = 1.0
        
        # Apply boost with tariff penalty
        for slot_idx in range(len(flows)):
            k_idx = self.action_slots["k"][slot_idx]
            edge_idx = self.action_slots["edge"][slot_idx]
            
            # Base boost from demand/backlog
            boost_factor = 1.0 + 0.18 * combined_pressure[k_idx]
            
            # Tariff penalty: reduce boost on high-tariff routes
            if tariff_obs[edge_idx, k_idx]:
                route_tariff = tariff[edge_idx, k_idx]
                if route_tariff > 1.5 * mean_tariff:
                    boost_factor *= 0.90  # 10% penalty for expensive routes
            
            flows[slot_idx] *= boost_factor
        
        # Soft cap
        allowed_capacity = self.capacity * action_mask
        total_allowed = np.sum(allowed_capacity)
        total_flows = np.sum(flows)
        if total_flows > total_allowed * 1.25:
            flows = flows * (total_allowed * 1.2 / (total_flows + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }