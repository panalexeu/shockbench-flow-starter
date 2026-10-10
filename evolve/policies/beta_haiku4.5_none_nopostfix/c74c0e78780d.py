# -2.1089137080091658
import numpy as np

class Agent:
    def __init__(self, config):
        self.config = config
        self.T = config['T']
        
        # Seed safely - handle seed as unsigned 32-bit
        seed = config['policy_seed']
        if isinstance(seed, (int, np.integer)):
            seed = int(seed) % (2**31)
            if seed < 0:
                seed = (-seed) % (2**31)
        else:
            seed = 0
        np.random.seed(seed)
        
        # Static structures (dicts of parallel lists)
        self.action_slots = config['static']['action_slots']
        self.override_slots = config['static']['override_slots']
        self.edges = config['static']['edges']
        self.commodities = config['static']['commodities']
        self.sinks = config['static']['sinks']
        
        # Layout structures (lists of tuples/indices)
        self.stock_slots = config['layout']['stock_slots']
        self.demands = config['layout']['demands']
        self.chokepoints = config['layout']['chokepoints']
        self.release_pairs = config['layout']['release_pairs']
        
        # Build reverse lookup maps
        self.stock_map = {}
        for i, slot in enumerate(self.stock_slots):
            key = tuple(slot) if isinstance(slot, list) else slot
            self.stock_map[key] = i
        
        self.demand_map = {}
        for i, slot in enumerate(self.demands):
            key = tuple(slot) if isinstance(slot, list) else slot
            self.demand_map[key] = i
        
        # Extract shortage penalties from sinks
        self.shortage_pi = {}
        for i, (sink_node, sink_k) in enumerate(self.demands):
            pi = 0.0
            for j in range(len(self.sinks['node'])):
                if self.sinks['node'][j] == sink_node and self.sinks['k'][j] == sink_k:
                    pi = float(self.sinks['pi'][j])
                    break
            self.shortage_pi[i] = pi
    
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
            
            # Get pipeline to understand in-transit goods
            pipeline_qty = observation['pipeline.qty']
            pipeline_arrival = observation['pipeline.arrival_week']
            pipeline_k = observation['pipeline.k']
            
            # Calculate in-transit quantities per (destination node, commodity)
            in_transit = {}
            for i, qty in enumerate(pipeline_qty):
                if pipeline_qty.observed[i] > 0 if hasattr(pipeline_qty, 'observed') else True:
                    arr_week = int(pipeline_arrival[i]) if i < len(pipeline_arrival) else week + 10
                    if arr_week <= week:
                        k = int(pipeline_k[i]) if i < len(pipeline_k) else 0
                        # This will arrive soon, factor into availability
                        # But we can't dispatch it until it arrives
            
            # Forecast horizon
            horizon = min(8, demand_forecast.shape[1] if demand_forecast.ndim > 1 else 1)
            
            # Calculate demand needs with better prioritization
            needs = []
            for d_idx in range(len(self.demands)):
                sink_node, sink_k = self.demands[d_idx]
                
                # Sum near-term demand (this + next week is most urgent)
                total_demand = 0.0
                for h in range(min(3, horizon)):  # Look at next 3 weeks
                    if demand_forecast.ndim > 1 and h < demand_forecast.shape[1]:
                        d_h = demand_forecast[d_idx, h]
                    else:
                        d_h = demand_forecast[d_idx] if h == 0 else 0.0
                    
                    if np.isfinite(d_h) and d_h > 0:
                        # Weight near weeks more heavily
                        weight = 3.0 - h  # Week 0: weight 3, week 1: weight 2, week 2: weight 1
                        total_demand += float(d_h) * weight
                
                # Account for backlog (unserved demand from prior weeks)
                bl = float(backlog[d_idx]) if np.isfinite(backlog[d_idx]) else 0.0
                net_need = max(0.0, total_demand - bl)
                
                if net_need > 0.01:
                    pi = self.shortage_pi.get(d_idx, 0.0)
                    # Prioritize by: shortage cost (descending), then urgency
                    priority = (-pi, -net_need, d_idx)
                    needs.append((priority, d_idx, net_need, sink_node, sink_k))
            
            needs.sort()
            
            # Track what we've allocated
            stock_used = {}
            edge_used = {}
            
            # Greedy allocation by priority
            for _, d_idx, need, sink_node, sink_k in needs:
                if need <= 0.01:
                    continue
                
                remaining = need
                
                # Find slots that can deliver this commodity, prefer shorter lead times
                candidates = []
                for slot_idx in range(num_action_slots):
                    # Check slot validity
                    if action_mask[slot_idx] == 0 or slot_mask[slot_idx] > 0:
                        continue
                    
                    edge_idx = self.action_slots['edge'][slot_idx]
                    slot_k = self.action_slots['k'][slot_idx]
                    
                    if slot_k != sink_k:
                        continue
                    
                    if graph_prohibited[edge_idx, sink_k] > 0:
                        continue
                    
                    lead_time = int(graph_tau[edge_idx])
                    candidates.append((lead_time, slot_idx))
                
                # Sort by lead time (shorter is better for this greedy allocation)
                candidates.sort()
                
                for _, slot_idx in candidates:
                    if remaining <= 0.01:
                        break
                    
                    edge_idx = self.action_slots['edge'][slot_idx]
                    slot_k = self.action_slots['k'][slot_idx]
                    edge_tail = self.edges['tail'][edge_idx]
                    edge_head = self.edges['head'][edge_idx]
                    
                    # Check stock at source
                    stock_key = tuple([edge_tail, sink_k])
                    if stock_key not in self.stock_map:
                        continue
                    
                    s_idx = self.stock_map[stock_key]
                    if stock_obs[s_idx] == 0:
                        continue
                    
                    avail = float(stock_qty[s_idx])
                    used = stock_used.get(s_idx, 0.0)
                    free_stock = max(0.0, avail - used)
                    
                    if free_stock <= 0.01:
                        continue
                    
                    # Edge capacity
                    cap = float(graph_u[edge_idx])
                    if not np.isfinite(cap) or cap < 0:
                        cap = 0.0
                    
                    edge_used_amt = edge_used.get(edge_idx, 0.0)
                    free_cap = max(0.0, cap - edge_used_amt)
                    
                    if free_cap <= 0.01:
                        continue
                    
                    # Apply chokepoint openness (multiplicative reduction)
                    effective_cap = free_cap
                    for choke_idx, choke_node in enumerate(self.chokepoints):
                        if edge_head == choke_node or edge_tail == choke_node:
                            if choke_idx < len(graph_open):
                                openness = float(graph_open[choke_idx])
                                if np.isfinite(openness) and openness >= 0:
                                    effective_cap *= openness
                    
                    if effective_cap <= 0.01:
                        continue
                    
                    # Allocate flow
                    flow = min(free_stock, effective_cap, remaining)
                    if flow > 0.01:
                        flows[slot_idx] = flow
                        remaining -= flow
                        stock_used[s_idx] = used + flow
                        edge_used[edge_idx] = edge_used_amt + flow
        
        except Exception:
            pass
        
        return {
            'flows': flows,
            'override_qty': override_qty,
            'release_mode': release_mode
        }