# 0.5602317842011287
import numpy as np

class Agent:
    def __init__(self, config=None):
        st = config['static']
        self.n_over = len(st['override_slots']['chokepoint'])
        self.n_rel = len(config['layout']['release_pairs'])
        ed = st['edges']
        self.tail = list(ed['tail'])
        self.head = list(ed['head'])
        self.u0 = np.array([np.inf if x is None else float(x) for x in ed['u0']], dtype=float)
        self.c0 = np.array([0.0 if x is None else float(x) for x in ed['c0']], dtype=float)
        self.v = [float(x) for x in st['commodities']['v']]
        self.se = list(st['action_slots']['edge'])
        self.sk = list(st['action_slots']['k'])
        self.sl = list(st['action_slots']['lane'])
        self.le = st['lanes']['edges']
        self.sidx = {(int(a), int(b)): r for r, (a, b) in enumerate(config['layout']['stock_slots'])}
        self.didx = {(int(a), int(b)): r for r, (a, b) in enumerate(config['layout']['demands'])}
        self.sedges = []
        self.last = []
        for i in range(len(self.se)):
            l = self.sl[i]
            es = [self.se[i]] if (l is None or not self.le[l]) else list(self.le[l])
            self.sedges.append(es)
            self.last.append(int(self.head[es[-1]]))
        self.prev_backlog = None

    def act(self, obs):
        u = np.array(obs['graph_now.u'], dtype=float)
        uo = obs.get('graph_now.u.observed')
        cap = np.where(np.array(uo) > 0, u, self.u0) if uo is not None else u
        cap = np.clip(np.where(np.isfinite(cap), cap, 0.0), 0, None)
        tar = np.array(obs['graph_now.tariff'], dtype=float)
        stock = np.clip(np.array(obs['stock.qty'], dtype=float), 0, None)
        bl = np.clip(np.array(obs['backlog.qty'], dtype=float), 0, None)
        fc = np.clip(np.array(obs['demand_forecast.qty'], dtype=float), 0, None)
        mask = np.array(obs['action_mask'], dtype=float)
        
        n = len(self.se)
        flows = np.zeros(n)
        cost = np.zeros(n)
        
        # Slot capacities and costs
        for i in range(n):
            es = self.sedges[i]
            flows[i] = max(min(cap[e] for e in es), 0.0)
            k = int(self.sk[i])
            cost[i] = sum(self.c0[e] for e in es) + 0.02 * tar[es[0], k] * self.v[k]
        
        # Pipeline inventory (including near-arrival shipments)
        it = {}
        if 'pipeline.qty' in obs:
            pq = np.array(obs['pipeline.qty'], dtype=float)
            pe = np.array(obs['pipeline.edge'], dtype=int)
            pk = np.array(obs['pipeline.k'], dtype=int)
            pa = np.array(obs['pipeline.arrival_week'], dtype=int)
            po = np.array(obs['pipeline.qty.observed']) if 'pipeline.qty.observed' in obs else np.ones_like(pq)
            week = int(obs['week'][0]) if 'week' in obs else 1
            for j in range(len(pq)):
                if po[j] > 0 and 0 <= pe[j] < len(self.head) and pa[j] <= week + 2:
                    key = (int(self.head[pe[j]]), int(pk[j]))
                    if key in self.didx:
                        it[key] = it.get(key, 0.0) + pq[j]
        
        # Detect demand spikes
        demand_spike_multiplier = 1.0
        if self.prev_backlog is not None:
            backlog_change = bl - self.prev_backlog
            if backlog_change.sum() > 2.0:
                demand_spike_multiplier = 1.5
        self.prev_backlog = bl.copy()
        
        # Compute need with adaptive weighting
        H = 7
        need = {}
        for key, d in self.didx.items():
            h = min(H, fc.shape[1])
            backlog_cost = bl[d] * 5.0 * demand_spike_multiplier
            forecast_cost = fc[d, :h].sum() * 1.3
            pipeline_credit = it.get(key, 0.0) * 0.8
            net = backlog_cost + forecast_cost - pipeline_credit
            need[key] = max(net, 0.0)
        
        # Cost-first greedy
        for i in sorted(range(n), key=lambda i: cost[i]):
            key = (self.last[i], int(self.sk[i]))
            if key in need and flows[i] > 0:
                q = min(flows[i], need[key])
                flows[i] = q
                need[key] = max(need[key] - q, 0.0)
        
        # Stock constraints
        groups = {}
        for i in range(n):
            key = (int(self.tail[self.se[i]]), int(self.sk[i]))
            if key in self.sidx:
                groups.setdefault(key, []).append(i)
        
        for key, idxs in groups.items():
            av = stock[self.sidx[key]]
            tot = sum(flows[i] for i in idxs)
            if tot > av + 1e-6 and tot > 0:
                s = av / tot
                for i in idxs:
                    flows[i] *= s
        
        flows = flows * mask
        return {'flows': flows, 'override_qty': np.zeros(self.n_over), 'release_mode': np.zeros(self.n_rel, dtype=np.int64)}