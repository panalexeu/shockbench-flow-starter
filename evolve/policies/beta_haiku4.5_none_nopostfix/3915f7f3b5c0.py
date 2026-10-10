# -2.1093559669226374
import numpy as np

class Agent:
    def __init__(self, config):
        self.config = config
        self.T = config['T']
        
        # Seed RNG safely
        seed = config['policy_seed']
        if isinstance(seed, int):
            seed = max(0, min(seed & 0x7FFFFFFF, 2**31 - 1))
        else:
            seed = 0
        self.rng = np.random.RandomState(seed)
        
        # Static tables: dicts of parallel lists
        self.action_slots = config['static']['action_slots']
        self.override_slots = config['static']['override_slots']
        self.edges = config['static']['edges']
        self.nodes = config['static']['nodes']
        self.commodities = config['static']['commodities']
        self.sinks = config['static']['sinks']
        
        # Layout tables: lists of index tuples
        self.stock_slots = config['layout']['stock_slots']
        self.supply_slots = config['layout']['supply_slots']
        self.demands = config['layout']['demands']
        self.chokepoints = config['layout']['chokepoints']
        self.release_pairs = config['layout']['release_pairs']
        self.cost_components = config['layout']['cost_components']
        
        # Build reverse lookup maps
        self.stock_map = {}
        for i, slot in enumerate(self.stock_slots):
            self.stock_map[tuple(slot)] = i
        
        self.demand_map = {}
        for i, slot in enumerate(self.demands):
            self.demand_map[tuple(slot)] = i
        
        # Build shortage cost map (pi values from sinks)
        self.shortage_cost = {}
        for i, (sink_node, sink_k) in enumerate(self.demands):
            # Find corresponding sink in sinks table
            for j in range(len(self.sinks['node'])):
                if self.sinks['node'][j] == sink_node and self.sinks['k'][j] == sink_k:
                    self.shortage_cost[i] = self.sinks['pi'][j]
                    break
            if i not in self.shortage_cost:
                self.shortage_cost[i] = 0
    
    def act(self, observation):
        week = int(observation['week'][0])
        
        # Get array sizes
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
            backlog_qty = observation['backlog.qty']
            demand_forecast = observation['demand_forecast.qty']
            graph_u = observation['graph_now.u']
            graph_prohibited = observation['graph_now.prohibited']
            graph_open = observation['graph_now.open']
            action_mask = observation['action_mask']
            slot_mask = observation['slot_mask']
            
            # Check for pending prohibitions that might warrant holding tanker cargo
            pending_prohibitions = observation.get('pending_prohibitions.edge')
            if pending_prohibitions is not None and pending_prohibitions.size > 0:
                # Hold tanker cargo if sanctions are imminent (within 2 weeks)
                for i in range(len(self.release_pairs)):
                    release_mode[i] = 2  # Hold by default when sanctions loom
            
            # Get demand for next few weeks
            if demand_forecast.ndim > 1:
                demand_t0 = demand_forecast[:, 0]
                demand_t1 = demand_forecast[:, 1] if demand_forecast.shape[1] > 1 else np.zeros(len(self.demands))
                demand_t2 = demand_forecast[:, 2] if demand_forecast.shape[1] > 2 else np.zeros(len(self.demands))
            else:
                demand_t0 = demand_forecast
                demand_t1 = np.zeros(len(self.demands))
                demand_t2 = np.zeros(len(self.demands))
            
            # Build list of demands sorted by shortage cost (highest first)
            demand_priorities = []
            for d_idx in range(len(self.demands)):
                d0 = demand_t0[d_idx] if np.isfinite(demand_t0[d_idx]) else 0
                d1 = demand_t1[d_idx] if np.isfinite(demand_t1[d_idx]) else 0
                d2 = demand_t2[d_idx] if np.isfinite(demand_t2[d_idx]) else 0
                current_backlog = backlog_qty[d_idx] if np.isfinite(backlog_qty[d_idx]) else 0
                
                # Total demand over next 3 weeks
                total_demand = max(0, d0 + d1 + d2 - current_backlog)
                if total_demand > 0.01:
                    shortage_cost = self.shortage_cost.get(d_idx, 0)
                    # Priority: higher cost first (use negative for sorting)
                    priority = (-shortage_cost, d_idx, total_demand)
                    demand_priorities.append(priority)
            
            # Sort by shortage cost (descending)
            demand_priorities.sort()
            
            # Greedy allocation of flows to satisfy high-priority demands
            edge_capacity_remaining = {i: float(graph_u[self.action_slots['edge'][i]]) for i in range(num_action_slots)}
            stock_allocated = {}
            
            for _, d_idx, total_needed in demand_priorities:
                if total_needed <= 0.01:
                    continue
                
                sink_node, sink_k = self.demands[d_idx]
                remaining_needed = total_needed
                
                # Try each action slot that can deliver this commodity
                for slot_idx in range(num_action_slots):
                    if remaining_needed <= 0.01:
                        break
                    
                    # Check if slot is available
                    if action_mask[slot_idx] == 0 or slot_mask[slot_idx] > 0:
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
                    
                    # Check stock at source
                    stock_key = (edge_tail, sink_k)
                    if stock_key not in self.stock_map:
                        continue
                    
                    stock_idx = self.stock_map[stock_key]
                    if stock_observed[stock_idx] == 0:
                        continue
                    
                    available = stock_qty[stock_idx]
                    if available <= 0.01:
                        continue
                    
                    # Get effective capacity
                    capacity = edge_capacity_remaining[slot_idx]
                    if not np.isfinite(capacity) or capacity <= 0:
                        continue
                    
                    # Apply chokepoint openness
                    for choke_idx, choke_node in enumerate(self.chokepoints):
                        if edge_head == choke_node or edge_tail == choke_node:
                            if choke_idx < len(graph_open):
                                openness = graph_open[choke_idx]
                                if np.isfinite(openness):
                                    capacity *= openness
                    
                    if capacity <= 0.01:
                        continue
                    
                    # Allocate flow
                    flow = min(available, capacity, remaining_needed)
                    if flow > 0.01:
                        flows[slot_idx] = flow
                        remaining_needed -= flow
                        edge_capacity_remaining[slot_idx] -= flow
        
        except Exception:
            pass
        
        return {
            'flows': flows,
            'override_qty': override_qty,
            'release_mode': release_mode
        }
