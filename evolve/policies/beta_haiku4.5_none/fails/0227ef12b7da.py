# -999
# error: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
import numpy as np

class Agent:
    """Pipeline-aware: send based on demand minus in-transit inventory."""
    
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
        self.commodities = static["commodities"]
        self.layout_demands = layout["demands"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = np.zeros_like(self.capacity)
        action_mask = observation["action_mask"]
        
        # Compute demand need per sink
        demand_forecast = observation["demand_forecast.qty"]
        backlog = observation["backlog.qty"]
        demand_forecast_obs = observation["demand_forecast.qty.observed"]
        backlog_obs = observation["backlog.qty.observed"]
        
        num_demands = len(self.layout_demands)
        sink_demand_need = np.zeros(num_demands)
        
        for sink_idx in range(num_demands):
            if backlog_obs[sink_idx]:
                sink_demand_need[sink_idx] += backlog[sink_idx]
            if demand_forecast_obs[sink_idx]:
                sink_demand_need[sink_idx] += np.sum(demand_forecast[sink_idx, :4])  # 4-week horizon
        
        # Estimate inventory already in pipeline to each sink
        pipeline_edge = observation["pipeline.edge"]
        pipeline_k = observation["pipeline.k"]
        pipeline_qty = observation["pipeline.qty"]
        pipeline_arrival = observation["pipeline.arrival_week"]
        pipeline_obs = observation["pipeline.qty.observed"]
        
        current_week = observation["week"][0]
        sink_pipeline_inventory = np.zeros(num_demands)
        
        for pipe_idx in range(len(pipeline_qty)):
            if not pipeline_obs[pipe_idx]:
                continue
            
            edge_idx = pipeline_edge[pipe_idx]
            k_idx = pipeline_k[pipe_idx]
            arrival_week = pipeline_arrival[pipe_idx]
            qty = pipeline_qty[pipe_idx]
            
            dest_node = self.edges["head"][edge_idx]
            
            # Assign pipeline qty to sinks it could serve
            for sink_idx in range(num_demands):
                sink_node = self.layout_demands[sink_idx][0]
                sink_k = self.layout_demands[sink_idx][1]
                if sink_k == k_idx and arrival_week <= current_week + 4:
                    sink_pipeline_inventory[sink_idx] += qty
        
        # Compute send intensity: respond to gap between need and pipeline
        max_need = np.max(sink_demand_need) if np.max(sink_demand_need) > 0 else 1.0
        sink_demand_need = sink_demand_need / max_need
        sink_pipeline_inventory = sink_pipeline_inventory / (max_need + 1e-6)
        
        # Send to close gaps: high send where need > pipeline
        sink_send_intensity = np.zeros(num_demands)
        for sink_idx in range(num_demands):
            gap = sink_demand_need[sink_idx] - 0.6 * sink_pipeline_inventory[sink_idx]
            sink_send_intensity[sink_idx] = np.clip(0.4 + 0.6 * gap, 0.2, 1.0)
        
        # Get cost info
        graph_c = observation["graph_now.c"]
        graph_c_obs = observation["graph_now.c.observed"]
        graph_tariff = observation["graph_now.tariff"]
        graph_tariff_obs = observation["graph_now.tariff.observed"]
        
        # Allocate flows
        for slot_idx in range(len(flows)):
            if not action_mask[slot_idx]:
                continue
            
            edge_idx = self.action_slots["edge"][slot_idx]
            k_idx = self.action_slots["k"][slot_idx]
            
            # Find max intensity among sinks this edge could serve
            max_intensity = 0.0
            for sink_idx in range(num_demands):
                if self.layout_demands[sink_idx][1] == k_idx:
                    max_intensity = max(max_intensity, sink_send_intensity[sink_idx])
            
            if max_intensity == 0.0:
                max_intensity = 0.5
            
            # Cost penalty
            cost_factor = 1.0
            if graph_c_obs[edge_idx]:
                c_val = graph_c[edge_idx]
                mean_cost = np.mean(graph_c[graph_c_obs.astype(bool)])
                if c_val > 1.3 * mean_cost:
                    cost_factor *= 0.9
            
            if graph_tariff_obs[edge_idx, k_idx]:
                tariff = graph_tariff[edge_idx, k_idx]
                if tariff > 0.10:  # 10% threshold
                    cost_factor *= (1.0 - min(0.15, tariff))
            
            flows[slot_idx] = self.capacity[slot_idx] * max_intensity * cost_factor
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }