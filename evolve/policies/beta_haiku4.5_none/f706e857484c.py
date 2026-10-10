# 0.4699062375097181
import numpy as np

class Agent:
    """Chokepoint-aware tanker override: release high urgency, hold when pipeline full."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        self.action_slots = action_slots
        self.sinks = static["sinks"]
        self.commodities = static["commodities"]
        self.edges = static["edges"]
        self.release_pairs = static.get("layout", {}).get("release_pairs", []) if isinstance(static.get("layout"), dict) else []
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = self.capacity.copy()
        action_mask = observation["action_mask"]
        flows = flows * action_mask
        
        # === REGULAR FLOW ALLOCATION ===
        backlog = observation["backlog.qty"]
        backlog_obs = observation["backlog.qty.observed"]
        num_commodities = len(self.commodities["id"])
        backlog_per_k = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(backlog_obs) and backlog_obs[sink_idx]:
                k_idx = self.sinks["k"][sink_idx]
                backlog_per_k[k_idx] += backlog[sink_idx]
        
        pipeline_qty = observation["pipeline.qty"]
        pipeline_k = observation["pipeline.k"]
        pipeline_obs = observation["pipeline.qty.observed"]
        
        pipeline_per_k = np.zeros(num_commodities)
        if pipeline_obs.any():
            for i in np.where(pipeline_obs)[0]:
                k_idx = int(pipeline_k[i])
                if k_idx < num_commodities:
                    pipeline_per_k[k_idx] += pipeline_qty[i]
        
        max_backlog = np.max(backlog_per_k) if np.max(backlog_per_k) > 0 else 1.0
        backlog_pressure = backlog_per_k / (max_backlog + 1e-9)
        
        max_pipeline = np.max(pipeline_per_k) if np.max(pipeline_per_k) > 0 else 1.0
        pipeline_fullness = pipeline_per_k / (max_pipeline + 1e-9)
        
        for slot_idx in range(len(flows)):
            k_idx = self.action_slots["k"][slot_idx]
            edge_idx = self.action_slots["edge"][slot_idx]
            
            # Aggressive backlog boost
            backlog_boost = 1.0 + 0.35 * backlog_pressure[k_idx]
            
            # Lead-time boost conditional on backlog + low pipeline
            lead_time_boost = 1.0
            if backlog_pressure[k_idx] > 0.25 and pipeline_fullness[k_idx] < 0.6:
                if edge_idx < len(self.edges["tau0"]):
                    tau = self.edges["tau0"][edge_idx]
                    if tau > 2:
                        tau_factor = min((tau - 1.0) / 5.0, 1.0)
                        lead_time_boost = 1.0 + 0.12 * tau_factor
            
            flows[slot_idx] *= (backlog_boost * lead_time_boost)
        
        # Hard capacity cap
        allowed_capacity = self.capacity * action_mask
        total_allowed = np.sum(allowed_capacity)
        total_flows = np.sum(flows)
        if total_flows > total_allowed * 1.2:
            flows = flows * (total_allowed * 1.2 / (total_flows + 1e-9))
        
        # === CHOKEPOINT OVERRIDE LOGIC ===
        # Only attempt override if release_pairs exist and masks are observed
        override_mask_obs = observation.get("override_mask.observed", np.array([0]))
        if override_mask_obs.any() > 0 and len(self.release_mode) > 0:
            override_mask = observation.get("override_mask", np.ones(len(self.override_qty), dtype=np.int8))
            
            # For each release pair, decide hold (2), override (1), or default (0)
            for pair_idx in range(len(self.release_mode)):
                if pair_idx < len(override_mask) and override_mask[pair_idx]:
                    # Find tanker commodity for this pair
                    # Release (override) if high backlog, hold if pipeline full
                    is_tanker = False
                    for comm_idx in range(num_commodities):
                        if self.commodities["override"][comm_idx] if "override" in self.commodities else False:
                            is_tanker = True
                            backlog_factor = backlog_pressure[comm_idx]
                            pipeline_factor = pipeline_fullness[comm_idx]
                            
                            if backlog_factor > 0.6 and pipeline_factor < 0.5:
                                # High backlog, low pipeline: release
                                self.release_mode[pair_idx] = 1
                            elif pipeline_factor > 0.8:
                                # Pipeline full: hold
                                self.release_mode[pair_idx] = 2
                            else:
                                # Default behavior
                                self.release_mode[pair_idx] = 0
                            break
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }