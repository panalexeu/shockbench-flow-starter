# -2.108566985675333
import numpy as np

class Agent:
    def __init__(self, config):
        self.config = config
        self.T = config['T']
        
        # Seed RNG
        seed = config['policy_seed']
        if isinstance(seed, int):
            np.random.seed(seed & ((1 << 31) - 1))
        else:
            np.random.seed(0)
        
        # Static structures: dicts of parallel lists
        self.action_slots = config['static']['action_slots']
        self.override_slots = config['static']['override_slots']
        self.edges = config['static']['edges']
        self.commodities = config['static']['commodities']
        
        # Layout structures: lists of index tuples
        self.stock_slots = config['layout']['stock_slots']
        self.demands = config['layout']['demands']
        self.chokepoints = config['layout']['chokepoints']
        self.release_pairs = config['layout']['release_pairs']
        
        # Build reverse lookup maps
        self.stock_map = {}
        for i, slot in enumerate(self.stock_slots):
            self.stock_map[tuple(slot)] = i
        
        self.demand_map = {}
        for i, slot in enumerate(self.demands):
            self.demand_map[tuple(slot)] = i
    
    def act(self, observation):
        try:
            week = int(observation['week'][0])
            
            # Get array sizes
            num_slots = len(self.action_slots['edge'])
            num_override = len(self.override_slots['chokepoint'])
            num_release = len(self.release_pairs)
            
            flows = np.zeros(num_slots, dtype=np.float64)
            override_qty = np.zeros(num_override, dtype=np.float64)
            release_mode = np.zeros(num_release, dtype=np.int64)
            
            # Extract observations
            stock_qty = observation['stock.qty']
            stock_obs = observation['stock.qty.observed']
            backlog_qty = observation['backlog.qty']
            demand_forecast = observation['demand_forecast.qty']
            graph_u = observation['graph_now.u']
            graph_prohibited = observation['graph_now.prohibited']
            graph_open = observation['graph_now.open']
            action_mask = observation['action_mask']
            
            # Get demand for next 2 weeks (lookahead)
            if demand_forecast.ndim > 1:
                demand_t0 = demand_forecast[:, 0]
                demand_t1 = demand_forecast[:, 1] if demand_forecast.shape[1] > 1 else np.zeros(len(self.demands))
            else:
                demand_t0 = demand_forecast
                demand_t1 = np.zeros(len(self.demands))
            
            # For each demand, calculate priority based on shortage cost and forecast
            demand_needs = []
            for d_idx in range(len(self.demands)):
                sink_node, sink_k = self.demands[d_idx]
                
                # Current and near-term demand
                d0 = demand_t0[d_idx] if np.isfinite(demand_t0[d_idx]) else 0
                d1 = demand_t1[d_idx] if np.isfinite(demand_t1[d_idx]) else 0
                current_backlog = backlog_qty[d_idx] if np.isfinite(backlog_qty[d_idx]) else 0
                
                # Total needed (this week + next week, minus backlog)
                total_need = max(0, d0 + d1 - current_backlog)
                if total_need > 0.01:
                    # Use commodity value as priority (higher value = higher priority)
                    priority = -self.commodities['v'][sink_k] if sink_k < len(self.commodities['v']) else 0
                    demand_needs.append((priority, d_idx, total_need, sink_node, sink_k))
            
            # Sort by priority (most valuable first)
            demand_needs.sort()
            
            # Greedily allocate stock to demands
            slot_capacity = {i: graph_u[self.action_slots['edge'][i]] 
                           for i in range(num_slots)}
            
            for priority, d_idx, needed, sink_node, sink_k in demand_needs:
                if needed <= 0.01:
                    continue
                
                remaining = needed
                
                # Find all slots that could deliver this commodity
                for slot_idx in range(num_slots):
                    if remaining <= 0.01:
                        break
                    
                    # Check slot viability
                    if action_mask[slot_idx] == 0:
                        continue
                    
                    edge_idx = self.action_slots['edge'][slot_idx]
                    slot_k = self.action_slots['k'][slot_idx]
                    
                    if slot_k != sink_k:
                        continue
                    
                    # Check prohibition
                    if graph_prohibited[edge_idx, sink_k] > 0:
                        continue
                    
                    edge_tail = self.edges['tail'][edge_idx]
                    edge_head = self.edges['head'][edge_idx]
                    
                    # Check if there's stock at source
                    stock_key = (edge_tail, sink_k)
                    if stock_key not in self.stock_map:
                        continue
                    
                    stock_idx = self.stock_map[stock_key]
                    if stock_obs[stock_idx] == 0:
                        continue
                    
                    available = stock_qty[stock_idx]
                    if available <= 0.01:
                        continue
                    
                    # Get effective capacity (account for chokepoint closure)
                    capacity = float(slot_capacity[slot_idx])
                    if not np.isfinite(capacity) or capacity <= 0:
                        continue
                    
                    # Apply chokepoint open fractions
                    for choke_idx, choke_node in enumerate(self.chokepoints):
                        if edge_head == choke_node or edge_tail == choke_node:
                            if choke_idx < len(graph_open):
                                openness = graph_open[choke_idx]
                                if np.isfinite(openness):
                                    capacity *= openness
                    
                    if capacity <= 0.01:
                        continue
                    
                    # Allocate flow
                    flow = min(available, capacity, remaining)
                    if flow > 0.01:
                        flows[slot_idx] = flow
                        remaining -= flow
                        slot_capacity[slot_idx] -= flow
        
        except Exception:
            pass
        
        return {
            'flows': flows,
            'override_qty': override_qty,
            'release_mode': release_mode
        }
