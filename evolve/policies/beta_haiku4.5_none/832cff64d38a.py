# -0.14087839642781322
import numpy as np

class Agent:
    """Dynamic demand-pipeline matching with measured urgency allocation."""
    
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
        self.commodities = static["commodities"]
        self.override_slots = static["override_slots"]
        self.layout_chokepoints = layout["chokepoints"]
        self.layout_release_pairs = layout["release_pairs"]
        self.layout_demands = layout["demands"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        # === DEMAND & URGENCY COMPUTATION ===
        backlog = observation["backlog.qty"]
        demand_forecast = observation["demand_forecast.qty"]
        num_demands = len(self.layout_demands)
        
        sink_urgency = np.zeros(num_demands)
        for sink_idx in range(num_demands):
            if sink_idx < len(backlog):
                sink_urgency[sink_idx] += backlog[sink_idx] * 2.0  # Backlog is more urgent
            if sink_idx < len(demand_forecast):
                sink_urgency[sink_idx] += np.sum(demand_forecast[sink_idx, :3])  # 3-week horizon
        
        # === PIPELINE INVENTORY ===
        pipeline_qty = observation["pipeline.qty"]
        pipeline_k = observation["pipeline.k"]
        pipeline_arrival = observation["pipeline.arrival_week"]
        pipeline_obs = observation["pipeline.qty.observed"]
        current_week = observation["week"][0]
        
        # Track inventory arriving in next 4 weeks per commodity
        num_commodities = len(self.commodities["id"])
        near_term_inventory = np.zeros(num_commodities)
        
        for pipe_idx in range(len(pipeline_qty)):
            if not pipeline_obs[pipe_idx]:
                continue
            k = int(pipeline_k[pipe_idx])
            arrival = int(pipeline_arrival[pipe_idx])
            if arrival <= current_week + 4:
                near_term_inventory[k] += pipeline_qty[pipe_idx]
        
        # === FLOW ALLOCATION ===
        flows = np.zeros_like(self.capacity)
        action_mask = observation["action_mask"]
        
        # Map demands to commodities
        sink_to_k = {}
        for sink_idx in range(num_demands):
            k = self.layout_demands[sink_idx][1]
            if k not in sink_to_k:
                sink_to_k[k] = []
            sink_to_k[k].append(sink_idx)
        
        # Per-commodity urgency: sum of all sinks
        commodity_urgency = np.zeros(num_commodities)
        for k, sink_indices in sink_to_k.items():
            commodity_urgency[k] = sum(sink_urgency[i] for i in sink_indices if i < len(sink_urgency))
        
        # Normalize
        max_urgency = np.max(commodity_urgency) if np.max(commodity_urgency) > 0 else 1.0
        commodity_urgency = commodity_urgency / max_urgency
        
        # Allocation intensity based on gap between urgency and pipeline
        intensity = np.zeros(num_commodities)
        for k in range(num_commodities):
            gap = commodity_urgency[k] - 0.3 * (near_term_inventory[k] / (max_urgency + 1e-6))
            intensity[k] = 0.5 + 0.5 * np.clip(gap, 0, 1)
        
        # Send flows proportional to intensity
        for slot_idx in range(len(flows)):
            if not action_mask[slot_idx]:
                continue
            k = self.action_slots["k"][slot_idx]
            flows[slot_idx] = self.capacity[slot_idx] * intensity[k]
        
        # === CHOKEPOINT OVERRIDES ===
        graph_open = observation["graph_now.open"]
        graph_kappa_tb = observation["graph_now.kappa.tb"]
        override_mask = observation["override_mask"]
        
        for override_idx in range(len(self.override_slots["chokepoint"])):
            if not override_mask[override_idx]:
                continue
            
            cp_idx = self.override_slots["chokepoint"][override_idx]
            k_idx = self.override_slots["k"][override_idx]
            
            # Deploy override if chokepoint is severely constrained and commodity is urgent
            is_constrained = (cp_idx < len(graph_open) and graph_open[cp_idx] < 0.6)
            throughput_low = (cp_idx < len(graph_kappa_tb) and graph_kappa_tb[cp_idx] < 0.7)
            is_urgent = intensity[k_idx] > 0.7
            
            if is_constrained and throughput_low and is_urgent:
                self.release_mode[override_idx] = 1
                self.override_qty[override_idx] = 40.0 * intensity[k_idx]
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }