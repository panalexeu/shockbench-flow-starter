# -2.1092247212644413
import numpy as np

class Agent:
    def __init__(self, config):
        self.config = config
        self.T = config['T']
        
        # Seed RNG safely
        seed = config['policy_seed']
        if isinstance(seed, int):
            seed = seed % (2**31)  # Keep in safe range
        np.random.seed(seed)
        
        # Static structures are dicts of parallel lists
        self.action_slots = config['static']['action_slots']
        self.override_slots = config['static']['override_slots']
        self.edges = config['static']['edges']
        self.nodes = config['static']['nodes']
        self.commodities = config['static']['commodities']
        
        # Layouts are lists
        self.stock_slots = config['layout']['stock_slots']
        self.supply_slots = config['layout']['supply_slots']
        self.demands = config['layout']['demands']
        self.chokepoints = config['layout']['chokepoints']
        self.release_pairs = config['layout']['release_pairs']
        
        # Build lookup maps
        self.stock_map = {}
        for i, slot in enumerate(self.stock_slots):
            self.stock_map[tuple(slot)] = i
        
        self.demand_map = {}
        for i, slot in enumerate(self.demands):
            self.demand_map[tuple(slot)] = i
    
    def act(self, observation):
        week = int(observation['week'][0])
        
        num_action_slots = len(self.action_slots['edge'])
        num_override_slots = len(self.override_slots['chokepoint'])
        num_release_pairs = len(self.release_pairs)
        
        flows = np.zeros(num_action_slots, dtype=np.float64)
        override_qty = np.zeros(num_override_slots, dtype=np.float64)
        release_mode = np.zeros(num_release_pairs, dtype=np.int64)
        
        try:
            # Extract observations
            stock_qty = observation['stock.qty']
            stock_observed = observation['stock.qty.observed']
            demand_forecast = observation['demand_forecast.qty']
            graph_capacity = observation['graph_now.u']
            graph_open = observation['graph_now.open']
            graph_prohibited = observation['graph_now.prohibited']
            action_mask = observation['action_mask']
            slot_mask = observation['slot_mask']
            backlog = observation['backlog.qty']
            
            # Get current and near-future demand (weeks 0 and 1 of forecast)
            demand_this_week = demand_forecast[:, 0] if demand_forecast.ndim > 1 else demand_forecast
            demand_next_week = demand_forecast[:, 1] if demand_forecast.ndim > 1 and demand_forecast.shape[1] > 1 else np.zeros(len(self.demands))
            
            # For each demand location, calculate required supply
            for demand_idx in range(len(self.demands)):
                # Current and future demand
                current_demand = demand_this_week[demand_idx] if np.isfinite(demand_this_week[demand_idx]) else 0
                future_demand = demand_next_week[demand_idx] if np.isfinite(demand_next_week[demand_idx]) else 0
                current_backlog = backlog[demand_idx] if np.isfinite(backlog[demand_idx]) else 0
                
                total_needed = max(0, current_demand + future_demand - current_backlog)
                if total_needed <= 0.01:
                    continue
                
                sink_node, sink_k = self.demands[demand_idx]
                
                # Try to route supply to this demand via available slots
                remaining = total_needed
                for slot_idx in range(num_action_slots):
                    if remaining <= 0.01:
                        break
                    
                    # Check slot availability
                    if action_mask[slot_idx] == 0:
                        continue
                    if slot_mask[slot_idx] > 0:
                        continue
                    
                    # Get slot info (parallel list access)
                    edge_idx = self.action_slots['edge'][slot_idx]
                    slot_k = self.action_slots['k'][slot_idx]
                    
                    if slot_k != sink_k:
                        continue
                    
                    # Check prohibition
                    if graph_prohibited[edge_idx, sink_k] > 0:
                        continue
                    
                    edge_head = self.edges['head'][edge_idx]
                    edge_tail = self.edges['tail'][edge_idx]
                    
                    # Check if there's stock at the source of this edge
                    stock_key = (edge_tail, sink_k)
                    if stock_key not in self.stock_map:
                        continue
                    
                    stock_idx = self.stock_map[stock_key]
                    if stock_observed[stock_idx] == 0:
                        continue
                    
                    available = stock_qty[stock_idx]
                    if available <= 0.01:
                        continue
                    
                    # Get capacity
                    capacity = graph_capacity[edge_idx]
                    if not np.isfinite(capacity):
                        capacity = 0
                    
                    # Apply chokepoint closure factor
                    for choke_idx, choke_node in enumerate(self.chokepoints):
                        if edge_head == choke_node or edge_tail == choke_node:
                            if graph_open.shape[0] > choke_idx:
                                openness = graph_open[choke_idx]
                                if np.isfinite(openness):
                                    capacity *= openness
                    
                    if capacity <= 0.01:
                        continue
                    
                    flow = min(available, capacity, remaining)
                    if flow > 0.01:
                        flows[slot_idx] = flow
                        remaining -= flow
        
        except Exception:
            # On any error, return zeros (naive strategy)
            pass
        
        return {
            'flows': flows,
            'override_qty': override_qty,
            'release_mode': release_mode
        }