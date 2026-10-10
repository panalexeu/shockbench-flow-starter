# 0.49608529913471106
import numpy as np

class Agent:
    def __init__(self, config):
        self.s = s = config['static']
        self.l = l = config['layout']
        self.T = int(config['T'])
        e = s['edges']
        self.tail = np.asarray(e['tail'], int)
        self.head = np.asarray(e['head'], int)
        self.u0 = np.asarray([0 if x is None else x for x in e['u0']], float)
        self.c0 = np.asarray([0 if x is None else x for x in e['c0']], float)
        self.t0 = np.asarray([1 if x is None else x for x in e['tau0']], float)
        self.v = np.asarray(s['commodities']['v'], float)
        self.pool = s['commodities']['pool']
        self.stock = [tuple(p) for p in l['stock_slots']]
        self.supply = [tuple(p) for p in l['supply_slots']]
        self.demands = [tuple(p) for p in l['demands']]
        self.di = {p:i for i,p in enumerate(self.demands)}
        self.cp = {n:i for i,n in enumerate(l['chokepoints'])}
        penalties = {(n,k):float(p) for n,k,p in zip(s['sinks']['node'],s['sinks']['k'],s['sinks']['pi'])}
        self.pi = [penalties.get(p,1.) for p in self.demands]
        self.maxpi = max(self.pi, default=1.)
        self.routes = []
        groups = {}
        a = s['action_slots']
        for i,(edge,k,lane) in enumerate(zip(a['edge'],a['k'],a['lane'])):
            lane = -1 if lane is None else int(lane)
            es = [int(edge)] if lane < 0 else list(s['lanes']['edges'][lane])
            cps = [] if lane < 0 else list(s['lanes']['chokepoints'][lane])
            src,dst = int(self.tail[es[0]]),int(self.head[es[-1]])
            self.routes.append((i,int(k),es,cps,src,dst))
            alt = e['alt_of'][edge] if lane < 0 else s['lanes']['alt_of'][lane]
            groups.setdefault((src,k,dst),[]).append((alt,min(self.u0[x] for x in es)))
        inc,out = {},{}
        for (src,k,dst),choices in groups.items():
            base = [cap for alt,cap in choices if alt is None]
            cap = sum(base) if base else max(cap for _,cap in choices)
            inc[(dst,k)] = inc.get((dst,k),0.)+cap
            out[(src,k)] = out.get((src,k),0.)+cap
        self.rate = {p:min(inc[p],out[p]) if p in inc and p in out else inc.get(p,out.get(p,0.)) for p in set(inc)|set(out)}
        self.memory = {}

    def field(self,o,key,default):
        default = np.asarray(default)
        x = np.asarray(o.get(key,default))
        mask = o.get(key+'.observed')
        return np.where(mask,x,default) if mask is not None and np.shape(mask)==x.shape else x.copy()

    def act(self,o):
        week = int(o['week'][0])
        ne,nk = len(self.tail),len(self.v)
        u = self.field(o,'graph_now.u',self.u0)
        c = self.field(o,'graph_now.c',self.c0)
        tau = np.maximum(1.,self.field(o,'graph_now.tau',self.t0))
        tariff = self.field(o,'graph_now.tariff',np.zeros((ne,nk)))
        banned = self.field(o,'graph_now.prohibited',np.zeros((ne,nk))).astype(bool)
        opened = self.field(o,'graph_now.open',np.ones(len(self.cp)))
        mask = np.asarray(o.get('action_mask',np.ones(len(self.routes))))
        available, projected = {},{}
        def add(p,t,q):
            if q > 0 and np.isfinite(q): projected.setdefault(p,[]).append((int(t),float(q)))
        stock = self.field(o,'stock.qty',np.zeros(len(self.stock)))
        for p,q in zip(self.stock,stock):
            available[p] = max(0.,float(q))
            add(p,week,q)
        supply = self.field(o,'graph_now.supply.avail',np.zeros(len(self.supply)))
        for p,q in zip(self.supply,supply): available[p] = available.get(p,0.)+max(0.,float(q))
        lanes = self.s['lanes']['edges']
        def endpoint(edge,lane):
            es = list(lanes[lane]) if 0 <= lane < len(lanes) else [edge]
            pos = es.index(edge) if edge in es else 0
            return int(self.head[es[-1]]),sum(tau[x] for x in es[pos+1:])
        q = np.asarray(o.get('pipeline.qty',[]))
        for j in np.flatnonzero(o.get('pipeline.qty.observed',np.ones(q.size))):
            if q[j] <= 0: continue
            edge = int(o['pipeline.edge'][j])
            if not 0 <= edge < ne: continue
            lane = int(o['pipeline.lane'][j]) if o.get('pipeline.lane.observed',np.ones(q.size))[j] else -1
            dst,rem = endpoint(edge,lane)
            add((dst,int(o['pipeline.k'][j])),max(week,int(o['pipeline.arrival_week'][j]))+int(np.ceil(rem)),q[j])
        q = np.asarray(o.get('wip.qty',[]))
        for j in np.flatnonzero(o.get('wip.qty.observed',np.ones(q.size))):
            add((int(o['wip.node'][j]),int(o['wip.k'][j])),max(week,int(o['wip.out_week'][j])),q[j])
        ends = {}
        for j in np.flatnonzero(o.get('closure_end.end_week.observed',[])):
            cp = int(o['closure_end.chokepoint'][j])
            ends[cp] = max(ends.get(cp,week),int(o['closure_end.end_week'][j])+1)
        lots = np.asarray(o.get('queue_lots.qty',np.zeros((0,0))))
        lm = np.asarray(o.get('queue_lots.qty.observed',np.ones(lots.shape)))
        for row,(cp,k,lane,edge) in enumerate(self.l.get('lot_keys',[])[:len(lots)]):
            lane = -1 if lane is None else int(lane)
            dst,rem = endpoint(int(edge),lane)
            release = week
            if cp in self.cp and opened[self.cp[cp]] <= .001: release = max(week,ends.get(cp,week+4))
            add((dst,k),release+int(np.ceil(tau[edge]+rem)),np.sum(np.where(lm[row],lots[row],0.)))
        pending = {}
        for j in np.flatnonzero(o.get('pending_prohibitions.edge.observed',[])):
            p = (int(o['pending_prohibitions.edge'][j]),int(o['pending_prohibitions.k'][j]))
            pending[p] = min(pending.get(p,self.T+1),int(o['pending_prohibitions.effective_week'][j]))
        forecast = np.asarray(o.get('demand_forecast.qty',np.zeros((len(self.demands),1))))
        backlog = self.field(o,'backlog.qty',np.zeros(len(self.demands)))
        def need(dst,k,lead):
            p = (dst,k)
            if p in self.di:
                d = self.di[p]
                h = min(self.T-week+1,max(1,int(np.ceil(lead))+2))
                row = forecast[d]
                n = min(h,len(row))
                target = max(0.,backlog[d])+np.sum(row[:n])
                if h > n and len(row): target += (h-n)*np.mean(row[-min(3,len(row)):])
                cutoff = week+h-1
            else:
                duration = min(max(2.,lead+2.),max(1.,self.T-week+1))
                target = self.rate.get(p,0.)*duration
                cutoff = week+int(np.ceil(duration))
            return max(0.,target-sum(q for t,q in projected.get(p,[]) if t <= cutoff))
        cp_left = {}
        for pool in ('tb','ct'):
            caps = self.field(o,'graph_now.kappa.'+pool,np.full(len(self.cp),np.inf))
            for cp,i in self.cp.items(): cp_left[(cp,pool)] = max(0.,caps[i]*opened[i])
        candidates = []
        for i,k,es,cps,src,dst in self.routes:
            if not mask[i] or any(banned[e,k] for e in es): continue
            if any(opened[self.cp[cp]] <= .001 for cp in cps if cp in self.cp): continue
            lead = sum(tau[e] for e in es)
            if week+lead > self.T+1: continue
            elapsed,illegal = 0.,False
            for e in es:
                if pending.get((e,k),self.T+1) <= week+elapsed: illegal = True
                elapsed += tau[e]
            if illegal: continue
            cap = min(max(0.,u[e]) for e in es)
            if cap <= 0 or need(dst,k,lead) <= 0: continue
            cost = sum(c[e]+tariff[e,k]*self.v[k] for e in es)
            priority = self.pi[self.di[(dst,k)]] if (dst,k) in self.di else self.maxpi*.25
            score = (cost+.002*self.v[k]*lead)/max(1.,priority)
            candidates.append((score,lead,i,cap))
        flows = np.zeros(len(self.routes))
        edge_left = np.maximum(0.,u.copy())
        for _,lead,i,cap in sorted(candidates):
            _,k,es,cps,src,dst = self.routes[i]
            q = min(cap,available.get((src,k),0.),need(dst,k,lead))
            for e in es: q = min(q,edge_left[e])
            for cp in cps: q = min(q,cp_left.get((cp,self.pool[k]),0.))
            if q <= 0 or not np.isfinite(q): continue
            flows[i] = q
            available[(src,k)] -= q
            for e in es: edge_left[e] -= q
            for cp in cps: cp_left[(cp,self.pool[k])] -= q
            add((dst,k),week+int(np.ceil(lead)),q)
        return {'flows':flows}
