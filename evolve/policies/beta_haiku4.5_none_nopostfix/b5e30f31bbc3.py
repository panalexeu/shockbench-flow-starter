# -2.1087729348498594
import numpy as np

class Agent:
    def __init__(self, config):
        self.config = config
        self.T = config['T']
        
        # Seed safely
        seed = config['policy_seed']
        if isinstance(seed, (int, np.integer)):
            seed = int(seed) % (2**31)
        else:
            seed = 42
        try:
            rng = np.random.RandomState(seed)
            self.rng = rng
        except:
            self.rng = np.random.RandomState(42)
        
        # Static structures (dicts of parallel lists)
        self.action_slots = config['static']['action_slots']
        self.override_slots = config['static']['override_slots']
        self.edges = config['static']['edges']
        self.commodities = config['static']['commodities']
        self.sinks = config['static']['sinks']
        
        # Layout structures
        self.stock_slots = config['layout']['stock_slots']
        self.demands = config['layout']['demands']
        self.chokepoints = config['layout']['chokepoints']
        self.release_pairs = config['layout']['release_pairs']
        
        # Build maps
        self.stock_map = {}
        for i, slot in enumerate(self.stock_slots):
            key = tuple(slot) if isinstance(slot, (list, tuple)) else slot
            self.stock_map[key] = i
        
        self.demand_map = {}
        for i, slot in enumerate(self.demands):
            key = tuple(slot) if isinstance(slot, (list, tuple)) else slot
            self.demand_map[key] = i
        
        # Extract shortage costs
        self.shortage_pi = {}
        for i, (sink_node, sink_k) in enumerate(self.demands):
            pi = 0.0
            for j in range(len(self.sinks['node'])):
                if self.sinks['node'][j] == sink_node and self.sinks['k'][j] == sink_k:
                    pi = float(self.sinks['pi'][j])
                    break
            self.shortage_pi[i] = pi
        
        # Extract commodity values for tiebreaking
        self.commodity_value = {}
        for k in range(len(self.commodities['id'])):
            self.commodity_value[k] = float(self.commodities['v'][k]) if k < len(self.commodities['v']) else 0.0
    
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
            stock_obs = observation['stock.qty.observed']
            backlog = observation['backlog.qty']
            demand_forecast = observation['demand_forecast.qty']
            graph_u = observation['graph_now.u']
            graph_tau = observation['graph_now.tau']
            graph_prohibited = observation['graph_now.prohibited']
            graph_open = observation['graph_now.open']
            action_mask = observation['action_mask']
            slot_mask = observation['slot_mask']
            pipeline_qty = observation['pipeline.qty']
            pipeline_arrival = observation['pipeline.arrival_week']
            
            # Estimate in-transit inventory per (destination, commodity)
            in_transit_qty = {}
            for i in range(len(pipeline_qty)):
                if pipeline_qty[i] > 0.01:
                    arr_week = int(pipeline_arrival[i]) if i < len(pipeline_arrival) else week + 100
                    if arr_week > week:  # Will arrive in future
                        # We can't use it now, so it doesn't help current demand
                        pass
            
            # Horizon for lookahead
            horizon = min(8, demand_forecast.shape[1] if demand_forecast.ndim > 1 else 1)
            
            # Build demand priority list
            # Priority: (1) backlog reduction, (2) lead-time-adjusted urgency, (3) shortage cost, (4) commodity value
            demand_priorities = []
            
            for d_idx in range(len(self.demands)):
                sink_node, sink_k = self.demands[d_idx]
                
                # Current backlog
                bl = float(backlog[d_idx]) if np.isfinite(backlog[d_idx]) else 0.0
                
                # Sum demand over next few weeks
                total_demand = 0.0
                for h in range(horizon):
                    if demand_forecast.ndim > 1 and h < demand_forecast.shape[1]:
                        d_h = demand_forecast[d_idx, h]
                    else:
                        d_h = demand_forecast[d_idx] if h == 0 else 0.0
                    
                    if np.isfinite(d_h) and d_h > 0:
                        total_demand += float(d_h)
                
                net_need = max(0.0, total_demand - bl)
                
                if net_need > 0.01 or bl > 0.01:
                    shortage_cost = self.shortage_pi.get(d_idx, 0.0)
                    commodity_val = self.commodity_value.get(sink_k, 0.0)
                    
                    # Priority tuple: higher shortage cost first, then higher backlog, then higher demand
                    priority = (-shortage_cost, -bl, -net_need, -commodity_val, d_idx)
                    demand_priorities.append((priority, d_idx, bl, net_need, sink_node, sink_k))
            
            demand_priorities.sort()
            
            # Track per-stock-source allocation
            stock_used = {}  # stock_idx -> total allocated
            edge_flow = {}   # edge_idx -> total allocated
            
            # Allocate by priority
            for _, d_idx, backlog_qty_val, net_demand, sink_node, sink_k in demand_priorities:
                # Prioritize backlog reduction
                needed = backlog_qty_val + net_demand
                if needed <= 0.01:
                    continue
                
                remaining = needed
                
                # Find all slots that deliver this commodity
                slot_candidates = []
                for slot_idx in range(num_action_slots):
                    if action_mask[slot_idx] == 0 or slot_mask[slot_idx] > 0:
                        continue
                    
                    edge_idx = self.action_slots['edge'][slot_idx]
                    slot_k = self.action_slots['k'][slot_idx]
                    
                    if slot_k != sink_k:
                        continue
                    
                    if graph_prohibited[edge_idx, sink_k] > 0:
                        continue
                    
                    lead_time = int(graph_tau[edge_idx])
                    slot_candidates.append((lead_time, slot_idx))
                
                # Process slots, preferring shorter lead times
                slot_candidates.sort()
                
                for _, slot_idx in slot_candidates:
                    if remaining <= 0.01:
                        break
                    
                    edge_idx = self.action_slots['edge'][slot_idx]
                    edge_tail = self.edges['tail'][edge_idx]
                    edge_head = self.edges['head'][edge_idx]
                    
                    # Check stock at source
                    stock_key = tuple([edge_tail, sink_k])
                    if stock_key not in self.stock_map:
                        continue
                    
                    stock_idx = self.stock_map[stock_key]
                    if stock_obs[stock_idx] == 0:
                        continue
                    
                    avail = float(stock_qty[stock_idx])
                    already_used = stock_used.get(stock_idx, 0.0)
                    free_stock = max(0.0, avail - already_used)
                    
                    if free_stock <= 0.01:
                        continue
                    
                    # Get capacity
                    cap = float(graph_u[edge_idx])
                    if not np.isfinite(cap) or cap < 0:
                        cap = 0.0
                    
                    already_edge = edge_flow.get(edge_idx, 0.0)
                    free_cap = max(0.0, cap - already_edge)
                    
                    if free_cap <= 0.01:
                        continue
                    
                    # Apply chokepoint openness (hard stops at 0, otherwise multiplicative)
                    effective_cap = free_cap
                    for choke_idx, choke_node in enumerate(self.chokepoints):
                        if edge_head == choke_node or edge_tail == choke_node:
                            if choke_idx < len(graph_open):
                                openness = float(graph_open[choke_idx])
                                if np.isfinite(openness):
                                    if openness <= 0:
                                        effective_cap = 0.0
                                        break
                                    else:
                                        effective_cap *= openness
                    
                    if effective_cap <= 0.01:
                        continue
                    
                    # Allocate flow
                    flow = min(free_stock, effective_cap, remaining)
                    if flow > 0.01:
                        flows[slot_idx] = flow
                        remaining -= flow
                        stock_used[stock_idx] = already_used + flow
                        edge_flow[edge_idx] = already_edge + flow
        
        except Exception:
            pass
        
        return {
            'flows': flows,
            'override_qty': override_qty,
            'release_mode': release_mode
        }
