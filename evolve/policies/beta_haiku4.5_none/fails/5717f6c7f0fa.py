# -999
# error: 'c'
import numpy as np

class Agent:
    """Hybrid: urgency × cost-efficiency × lead-time boost, with disruption adjustment."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        self.action_slots = action_slots
        self.edges = static["edges"]
        self.commodities = static["commodities"]
        self.sinks = static["sinks"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = np.zeros_like(self.capacity)
        action_mask = observation["action_mask"]
        week = int(observation["week"][0])
        
        # Demand and inventory signals
        demand_forecast = observation["demand_forecast.qty"]
        backlog = observation["backlog.qty"]
        stock = observation["stock.qty"]
        
        # Compute commodity urgency
        num_commodities = len(self.commodities["id"])
        commodity_urgency = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            k = self.sinks["k"][sink_idx]
            if sink_idx < len(demand_forecast):
                # Weight weeks 0-2 heavily for near-term urgency
                commodity_urgency[k] += demand_forecast[sink_idx, 0] * 1.5 + np.sum(demand_forecast[sink_idx, 1:3])
            if sink_idx < len(backlog):
                commodity_urgency[k] += backlog[sink_idx] * 2.5
        
        max_urgency = np.max(commodity_urgency) if np.max(commodity_urgency) > 0 else 1.0
        commodity_urgency = commodity_urgency / (max_urgency + 1e-9)
        
        # Cost and lead-time data
        edge_costs = np.array(self.edges["c"], dtype=float)
        edge_tau = np.array(self.edges["tau0"], dtype=float)
        commodity_value = np.array(self.commodities["v"], dtype=float)
        
        # Disruption signals
        warning_score = observation["warning.score"]
        max_warning = np.max(warning_score) if warning_score.size > 0 else 0.0
        graph_open = observation["graph_now.open"]
        min_open = np.min(graph_open) if graph_open.size > 0 else 1.0
        
        disruption_level = max(max_warning * 0.3, (1.0 - min_open) * 0.4)
        
        # Last-week performance
        last_served = observation["last_week.sinks.served"]
        last_demand = observation["last_week.sinks.demand"]
        shortage_rate = 1.0 - (np.sum(last_served) / (np.sum(last_demand) + 1e-6))
        
        # Allocation for each route
        for slot_idx in range(len(flows)):
            if not action_mask[slot_idx]:
                continue
            
            edge_idx = self.action_slots["edge"][slot_idx]
            k = self.action_slots["k"][slot_idx]
            
            # Cost efficiency (cheaper routes preferred)
            edge_cost = edge_costs[edge_idx] if edge_costs[edge_idx] > 0 else 1.0
            cost_efficiency = 1.0 / (edge_cost + 1e-6)
            cost_efficiency = min(cost_efficiency, 2.0)  # Cap to avoid extreme values
            
            # Lead-time boost (longer routes front-loaded)
            lead_time = edge_tau[edge_idx] if edge_tau[edge_idx] > 0 else 1.0
            lead_boost = 1.0 + 0.12 * (lead_time / 6.0)
            
            # Value factor: higher-value commodities prioritized in shortage
            value_factor = commodity_value[k] / (np.max(commodity_value) + 1e-9)
            
            # Urgency × value weight
            urgency_weight = 0.5 * commodity_urgency[k] + 0.3 * value_factor
            
            # Base allocation: 0.6 to 1.2 depending on route quality and demand
            base_allocation = 0.6 + 0.4 * urgency_weight
            
            # Cost efficiency modifier: boost cheap routes, reduce expensive ones
            cost_mod = 0.9 + 0.2 * min(cost_efficiency / 2.0, 1.0)
            
            # Lead-time pre-buffering
            lead_mod = lead_boost
            
            # Disruption response: boost when high-warning or bottleneck closure
            disruption_mod = 1.0 + 0.25 * disruption_level
            
            # Shortage response: increase when last-week shortage high
            shortage_mod = 1.0 + 0.2 * min(shortage_rate, 1.0)
            
            # Combined factor
            allocation_factor = base_allocation * cost_mod * lead_mod * disruption_mod * shortage_mod
            allocation_factor = min(allocation_factor, 1.4)  # Hard cap at 1.4x
            allocation_factor = max(allocation_factor, 0.5)  # Floor at 0.5x to ensure some baseline
            
            flows[slot_idx] = self.capacity[slot_idx] * allocation_factor
        
        # Soft normalization: if total flows exceed allowed, scale proportionally
        allowed_capacity = self.capacity * action_mask
        total_allowed = np.sum(allowed_capacity)
        total_flows = np.sum(flows)
        
        if total_allowed > 0 and total_flows > total_allowed * 1.2:
            scale_factor = (total_allowed * 1.1) / (total_flows + 1e-9)
            flows = flows * scale_factor
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }
