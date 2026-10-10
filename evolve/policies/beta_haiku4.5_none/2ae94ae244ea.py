# 0.006889424796628428
import numpy as np

class Agent:
    """Hybrid demand-cost-inventory model: balanced allocation."""
    
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
        self.layout = layout
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = np.zeros_like(self.capacity)
        action_mask = observation["action_mask"]
        
        # Demand signals
        demand_forecast = observation["demand_forecast.qty"]
        backlog = observation["backlog.qty"]
        stock = observation["stock.qty"]
        stock_observed = observation["stock.qty.observed"]
        last_served = observation["last_week.sinks.served"]
        last_demand = observation["last_week.sinks.demand"]
        
        # Compute commodity urgency: backlog + forecast demand
        num_commodities = len(self.commodities["id"])
        urgency = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            k = self.sinks["k"][sink_idx]
            if sink_idx < len(backlog):
                urgency[k] += backlog[sink_idx] * 5.0  # Backlog is high priority
            if sink_idx < len(demand_forecast):
                # Current and near-term demand
                urgency[k] += demand_forecast[sink_idx, 0] * 1.0
                if len(demand_forecast[sink_idx]) > 1:
                    urgency[k] += demand_forecast[sink_idx, 1] * 0.5
        
        # Shortage rate from last week
        shortage_rate = 1.0 - (np.sum(last_served) / (np.sum(last_demand) + 1e-6))
        
        # Cost and tariff data
        graph_c = observation["graph_now.c"]
        graph_tariff = observation["graph_now.tariff"]
        tariff_observed = observation["graph_now.tariff.observed"]
        
        # Compute cost factor for each slot
        cost_factor_by_slot = np.ones(len(flows))
        if len(graph_c) > 0:
            cost_median = np.median(graph_c)
            cost_p75 = np.percentile(graph_c, 75)
            
            for slot_idx in range(len(flows)):
                edge_idx = self.action_slots["edge"][slot_idx]
                k_idx = self.action_slots["k"][slot_idx]
                
                if edge_idx < len(graph_c):
                    edge_cost = float(graph_c[edge_idx])
                    if edge_cost > cost_p75:
                        # Reduce by 10-20% on expensive routes
                        cost_factor_by_slot[slot_idx] *= 0.85
                
                # Check tariff
                if tariff_observed.sum() > 0 and edge_idx < len(graph_tariff) and k_idx < len(graph_tariff[0]):
                    tariff_rate = float(graph_tariff[edge_idx, k_idx])
                    if tariff_rate > 0.1:  # >10% tariff
                        cost_factor_by_slot[slot_idx] *= (1.0 - min(0.15, tariff_rate * 0.5))
        
        # Inventory check: reduce if destination has excess stock
        stock_by_node_k = {}
        if stock_observed.sum() > 0:
            for stock_slot_idx in range(len(self.layout["stock_slots"])):
                node_k = tuple(self.layout["stock_slots"][stock_slot_idx])
                if stock_observed[stock_slot_idx]:
                    stock_by_node_k[node_k] = float(stock[stock_slot_idx])
        
        avg_stock = np.mean(list(stock_by_node_k.values())) if stock_by_node_k else 0.0
        high_stock_threshold = max(avg_stock * 1.5, 1.0)
        
        # Allocate flows
        total_urgency = np.sum(urgency) + 1e-6
        
        for slot_idx in range(len(flows)):
            if not action_mask[slot_idx]:
                continue
            
            k_idx = self.action_slots["k"][slot_idx]
            edge_idx = self.action_slots["edge"][slot_idx]
            
            # Base allocation from urgency
            urgency_weight = urgency[k_idx] / total_urgency if total_urgency > 0 else 1.0 / num_commodities
            base_factor = 0.6 + 0.4 * min(urgency_weight * 10.0, 1.0)
            
            # Shortage boost
            shortage_boost = 1.0 + shortage_rate * 0.3
            
            # Cost reduction
            cost_adjust = cost_factor_by_slot[slot_idx]
            
            # Stock penalty: if destination has excess, reduce send
            stock_penalty = 1.0
            if edge_idx < len(self.edges["head"]):
                dest_node = int(self.edges["head"][edge_idx])
                node_k_key = (dest_node, k_idx)
                if node_k_key in stock_by_node_k:
                    dest_stock = stock_by_node_k[node_k_key]
                    if dest_stock > high_stock_threshold:
                        stock_penalty = 0.7
            
            allocation_factor = base_factor * shortage_boost * cost_adjust * stock_penalty
            flows[slot_idx] = self.capacity[slot_idx] * min(1.2, allocation_factor)
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }