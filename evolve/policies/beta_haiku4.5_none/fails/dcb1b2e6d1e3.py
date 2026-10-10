# -999
# error: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
import numpy as np

class Agent:
    """Integrated strategy: demand-pipeline balance + chokepoint override + risk awareness."""
    
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
        self.layout_demands = layout["demands"]
        self.layout_chokepoints = layout["chokepoints"]
        self.layout_release_pairs = layout["release_pairs"]
        self.override_slots = static["override_slots"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = np.zeros_like(self.capacity)
        action_mask = observation["action_mask"]
        
        # === DEMAND ASSESSMENT ===
        demand_forecast = observation["demand_forecast.qty"]
        backlog = observation["backlog.qty"]
        demand_forecast_obs = observation["demand_forecast.qty.observed"]
        backlog_obs = observation["backlog.qty.observed"]
        
        num_demands = len(self.layout_demands)
        sink_need = np.zeros(num_demands)
        
        for sink_idx in range(num_demands):
            if backlog_obs[sink_idx]:
                sink_need[sink_idx] += backlog[sink_idx] * 1.5
            if demand_forecast_obs[sink_idx]:
                sink_need[sink_idx] += np.sum(demand_forecast[sink_idx, :3])
        
        # === PIPELINE INVENTORY ===
        pipeline_edge = observation["pipeline.edge"]
        pipeline_k = observation["pipeline.k"]
        pipeline_qty = observation["pipeline.qty"]
        pipeline_arrival = observation["pipeline.arrival_week"]
        pipeline_obs = observation["pipeline.qty.observed"]
        
        current_week = observation["week"][0]
        sink_pipeline = np.zeros(num_demands)
        
        for pipe_idx in range(len(pipeline_qty)):
            if not pipeline_obs[pipe_idx]:
                continue
            k_idx = pipeline_k[pipe_idx]
            qty = pipeline_qty[pipe_idx]
            arrival_week = pipeline_arrival[pipe_idx]
            
            for sink_idx in range(num_demands):
                if self.layout_demands[sink_idx][1] == k_idx and arrival_week <= current_week + 3:
                    sink_pipeline[sink_idx] += qty
        
        # Normalize
        max_need = np.max(sink_need) if np.max(sink_need) > 0 else 1.0
        sink_need = sink_need / max_need
        sink_pipeline = sink_pipeline / (max_need + 1e-6)
        
        # Send intensity: cover gap
        sink_intensity = np.zeros(num_demands)
        for sink_idx in range(num_demands):
            gap = max(0, sink_need[sink_idx] - 0.5 * sink_pipeline[sink_idx])
            sink_intensity[sink_idx] = 0.3 + 0.7 * np.clip(gap, 0, 1)
        
        # === COST ASSESSMENT ===
        graph_c = observation["graph_now.c"]
        graph_c_obs = observation["graph_now.c.observed"]
        graph_tariff = observation["graph_now.tariff"]
        graph_tariff_obs = observation["graph_now.tariff.observed"]
        
        mean_cost = np.mean(graph_c[graph_c_obs.astype(bool)]) if graph_c_obs.any() else 1.0
        
        # === WAR RISK ASSESSMENT ===
        graph_war_risk = observation["graph_now.war_risk"]
        
        # === CHOKEPOINT & QUEUE STATUS ===
        graph_open = observation["graph_now.open"]
        queue_lots = observation["queue_lots.qty"]
        queue_lots_obs = observation["queue_lots.qty.observed"]
        
        # Identify constrained chokepoints
        constrained_chokepoints = set()
        for cp_idx in range(len(self.layout_chokepoints)):
            if graph_open[cp_idx] < 0.5:  # Less than 50% open
                constrained_chokepoints.add(self.layout_chokepoints[cp_idx])
        
        # === FLOW ALLOCATION ===
        for slot_idx in range(len(flows)):
            if not action_mask[slot_idx]:
                continue
            
            edge_idx = self.action_slots["edge"][slot_idx]
            k_idx = self.action_slots["k"][slot_idx]
            
            # Find max intensity for this commodity
            max_intensity = 0.0
            for sink_idx in range(num_demands):
                if self.layout_demands[sink_idx][1] == k_idx:
                    max_intensity = max(max_intensity, sink_intensity[sink_idx])
            
            if max_intensity == 0.0:
                max_intensity = 0.4
            
            # Cost penalty
            cost_factor = 1.0
            if graph_c_obs[edge_idx]:
                if graph_c[edge_idx] > 1.5 * mean_cost:
                    cost_factor *= 0.8
            
            if graph_tariff_obs[edge_idx, k_idx]:
                tariff = graph_tariff[edge_idx, k_idx]
                if tariff > 0.12:
                    cost_factor *= max(0.7, 1.0 - tariff)
            
            # War risk penalty
            dest_node = self.edges["head"][edge_idx]
            if dest_node in constrained_chokepoints:
                cp_idx = self.layout_chokepoints.index(dest_node)
                if graph_war_risk[cp_idx] > 0:
                    cost_factor *= 0.85
            
            flows[slot_idx] = self.capacity[slot_idx] * max_intensity * cost_factor
        
        # === CHOKEPOINT OVERRIDES ===
        # For tanker commodities at high-urgency chokepoints, use override
        override_mask = observation["override_mask"]
        
        for override_idx in range(len(self.override_slots["chokepoint"])):
            if not override_mask[override_idx]:
                continue
            
            cp_node = self.override_slots["chokepoint"][override_idx]
            k_idx = self.override_slots["k"][override_idx]
            
            # Find urgency for this commodity
            max_urgency = 0.0
            for sink_idx in range(num_demands):
                if self.layout_demands[sink_idx][1] == k_idx:
                    max_urgency = max(max_urgency, sink_intensity[sink_idx])
            
            # Use override if urgency is high (>0.7) and chokepoint is constrained
            if max_urgency > 0.7 and cp_node in constrained_chokepoints:
                cp_idx = self.layout_chokepoints.index(cp_node)
                if graph_open[cp_idx] < 0.6:
                    self.release_mode[self.layout_release_pairs.index([cp_node, k_idx])] = 1
                    self.override_qty[override_idx] = 0.8 * self.capacity[min(override_idx, len(self.capacity) - 1)]
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }