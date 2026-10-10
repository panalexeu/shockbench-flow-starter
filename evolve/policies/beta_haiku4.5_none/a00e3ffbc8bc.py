# 0.3922465973614549
import numpy as np

class Agent:
    """Lead-time aware: front-load high-lead routes to buffer disruptions."""
    
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
        
        # Get demand forecast and backlog
        demand_forecast = observation["demand_forecast.qty"]
        backlog = observation["backlog.qty"]
        stock = observation["stock.qty"]
        pipeline_qty = observation["pipeline.qty"]
        
        # Compute commodity demand urgency
        num_commodities = len(self.commodities["id"])
        commodity_urgency = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            k = self.sinks["k"][sink_idx]
            if sink_idx < len(demand_forecast):
                # Weight recent demand more heavily
                commodity_urgency[k] += demand_forecast[sink_idx, 0] * 2.0 + np.sum(demand_forecast[sink_idx, 1:4])
            if sink_idx < len(backlog):
                commodity_urgency[k] += backlog[sink_idx] * 3.0
        
        # Normalize
        max_urgency = np.max(commodity_urgency) if np.max(commodity_urgency) > 0 else 1.0
        commodity_urgency = commodity_urgency / (max_urgency + 1e-9)
        
        # Get lead times
        edge_tau = np.array(self.edges["tau0"], dtype=float)
        
        # Check for disruption warnings and threats
        warning_score = observation["warning.score"]
        max_warning = np.max(warning_score) if warning_score.size > 0 else 0.0
        graph_open = observation["graph_now.open"]
        min_open = np.min(graph_open) if graph_open.size > 0 else 1.0
        
        disruption_level = max(max_warning * 0.4, (1.0 - min_open) * 0.6)
        
        # Allocation strategy
        for slot_idx in range(len(flows)):
            if not action_mask[slot_idx]:
                continue
            
            edge_idx = self.action_slots["edge"][slot_idx]
            k = self.action_slots["k"][slot_idx]
            
            # Lead-time factor: longer routes get boosted to pre-buffer
            lead_time = edge_tau[edge_idx] if edge_tau[edge_idx] > 0 else 1.0
            lead_time_boost = min(1.0 + 0.15 * (lead_time / 6.0), 1.5)  # Up to 1.5x for long routes
            
            # Disruption factor: boost when signals are high
            disruption_boost = 1.0 + 0.3 * disruption_level
            
            # Urgency factor
            urgency_factor = 0.7 + 0.3 * commodity_urgency[k]
            
            # Combined allocation
            allocation_factor = urgency_factor * lead_time_boost * disruption_boost
            allocation_factor = min(allocation_factor, 1.5)  # Cap at 1.5x
            
            flows[slot_idx] = self.capacity[slot_idx] * allocation_factor
        
        # Normalize if total exceeds a reasonable bound
        allowed_capacity = self.capacity * action_mask
        total_allowed = np.sum(allowed_capacity)
        if total_allowed > 0 and np.sum(flows) > total_allowed * 1.3:
            flows = flows * (total_allowed * 1.2 / (np.sum(flows) + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }
