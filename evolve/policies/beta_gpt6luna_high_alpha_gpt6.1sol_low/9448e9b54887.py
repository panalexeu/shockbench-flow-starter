# 0.49767556783699146
import numpy as np

class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.T = int(config['T'])
        s = self.s
        e = s['edges']
        self.tail = np.asarray(e['tail'], int)
        self.head = np.asarray(e['head'], int)
        self.u0 = np.array([0 if x is None else x for x in e['u0']], float)
        self.c0 = np.array([0 if x is None else x for x in e['c0']], float)
        self.t0 = np.array([1 if x is None else x for x in e['tau0']], float)
        self.v = np.asarray(s['commodities']['v'], float)
        self.pool = s['commodities']['pool']
        self.stocks = [tuple(map(int, p)) for p in self.l['stock_slots']]
        self.supplies = [tuple(map(int, p)) for p in self.l['supply_slots']]
        self.demands = [tuple(map(int, p)) for p in self.l['demands']]
        self.di = {p:i for i,p in enumerate(self.demands)}
        self.cp = {int(n):i for i,n in enumerate(self.l['chokepoints'])}
        sink_map = {(int(n),int(k)):i for i,(n,k) in enumerate(zip(s['sinks']['node'],s['sinks']['k']))}
        self.pi = {}
        self.carry = {}
        for p in self.demands:
            j = sink_map[p]
            self.pi[p] = float(s['sinks']['pi'][j])
            self.carry[p] = bool(s['sinks']['backlog'][j])
        self.maxpi = max([1.] + list(self.pi.values()))
        self.routes = []
        grouped = {}
        a = s['action_slots']
        for i,(edge,k,lane) in enumerate(zip(a['edge'],a['k'],a['lane'])):
            edge,k = int(edge),int(k)
            lane = -1 if lane is None else int(lane)
            es = [edge] if lane < 0 else list(map(int,s['lanes']['edges'][lane]))
            cps = [] if lane < 0 else list(map(int,s['lanes']['chokepoints'][lane]))
            src,dst = int(self.tail[es[0]]),int(self.head[es[-1]])
            self.routes.append((i,k,lane,es,cps,(src,k),(dst,k)))
            alt = e['alt_of'][edge] if lane < 0 else s['lanes']['alt_of'][lane]
            grouped.setdefault((src,k,dst),[]).append((alt,min(self.u0[x] for x in es)))
        incoming,outgoing = {},{}
        for (src,k,dst),choices in grouped.items():
            base = [q for alt,q in choices if alt is None]
            q = sum(base) if base else max(q for _,q in choices)
            incoming[(dst,k)] = incoming.get((dst,k),0.) + q
            outgoing[(src,k)] = outgoing.get((src,k),0.) + q
        self.rate = {}
        for p in set(incoming)|set(outgoing):
            x,y = incoming.get(p),outgoing.get(p)
            self.rate[p] = min(x,y) if x is not None and y is not None else max(x or 0.,y or 0.)
        self.priority = dict(self.pi)
        self.depth = {p:0 for p in self.demands}
        for _ in range(len(s['nodes']['id'])):
            changed = False
            for _,k,_,es,cps,src,dst in self.routes:
                if dst in self.depth:
                    d = self.depth[dst]+1
                    if src not in self.depth or d < self.depth[src]:
                        self.depth[src] = d
                        changed = True
                    value = self.priority.get(dst,1.) / 1.12
                    self.priority[src] = max(self.priority.get(src,0.),value)
            if not changed:
                break
        self.memory = {}

    def field(self,o,name,default,remember=False):
        default = np.asarray(default)
        fallback = self.memory.get(name,default) if remember else default
        if name not in o:
            return fallback.copy()
        x = np.asarray(o[name])
        m = o.get(name+'.observed')
        if m is not None and np.shape(m) == x.shape:
            x = np.where(m,x,fallback)
        if remember:
            self.memory[name] = x.copy()
        return x

    def act(self,o):
        week = int(np.asarray(o['week']).flat[0])
        ne,nk = len(self.tail),len(self.v)
        u = self.field(o,'graph_now.u',self.u0,True).astype(float)
        c = self.field(o,'graph_now.c',self.c0,True)
        tau = self.field(o,'graph_now.tau',self.t0,True)
        tariff = self.field(o,'graph_now.tariff',np.zeros((ne,nk)),True)
        banned = self.field(o,'graph_now.prohibited',np.zeros((ne,nk)),True).astype(bool)
        opened = self.field(o,'graph_now.open',np.ones(len(self.cp)),True)
        mask = np.asarray(o.get('action_mask',np.ones(len(self.routes))))
        stock = self.field(o,'stock.qty',np.zeros(len(self.stocks)))
        remaining = {p:max(0.,float(stock[i])) for i,p in enumerate(self.stocks)}
        available = dict(remaining)
        supply = self.field(o,'graph_now.supply.avail',np.zeros(len(self.supplies)),True)
        for i,p in enumerate(self.supplies):
            available[p] = available.get(p,0.) + max(0.,float(supply[i]))
        projected = {}
        def add(p,t,q):
            if q > 0:
                projected.setdefault(p,[]).append((int(t),float(q)))
        def destination(edge,lane,start):
            dst = int(self.head[edge])
            if lane is not None and 0 <= int(lane) < len(self.s['lanes']['edges']):
                es = self.s['lanes']['edges'][int(lane)]
                dst = int(self.head[es[-1]])
                if edge in es:
                    start += sum(max(1.,tau[x]) for x in es[es.index(edge)+1:])
            return dst,int(np.ceil(start))
        q = np.asarray(o.get('pipeline.qty',[]))
        live = np.asarray(o.get('pipeline.qty.observed',np.ones(q.shape)),bool)
        for j in np.flatnonzero(live):
            edge = int(o['pipeline.edge'][j])
            if not 0 <= edge < ne:
                continue
            lane = int(o['pipeline.lane'][j]) if o.get('pipeline.lane.observed',np.ones(q.shape))[j] else -1
            dst,arrival = destination(edge,lane,int(o['pipeline.arrival_week'][j]))
            add((dst,int(o['pipeline.k'][j])),max(week,arrival),q[j])
        q = np.asarray(o.get('wip.qty',[]))
        for j in np.flatnonzero(o.get('wip.qty.observed',np.ones(q.shape))):
            add((int(o['wip.node'][j]),int(o['wip.k'][j])),max(week,int(o['wip.out_week'][j])),q[j])
        closure = {}
        for j in np.flatnonzero(o.get('closure_end.chokepoint.observed',[])):
            if o.get('closure_end.end_week.observed',[])[j]:
                node = int(o['closure_end.chokepoint'][j])
                closure[node] = max(closure.get(node,week),int(o['closure_end.end_week'][j])+1)
        lots = np.asarray(o.get('queue_lots.qty',[]))
        if lots.ndim == 2:
            lm = np.asarray(o.get('queue_lots.qty.observed',np.ones(lots.shape)))
            for row,(cp,k,lane,edge) in enumerate(self.l.get('lot_keys',[])):
                if row >= lots.shape[0]:
                    break
                edge,cp,k = int(edge),int(cp),int(k)
                release = week
                if cp in self.cp and opened[self.cp[cp]] < .001:
                    release = closure.get(cp,week+8)
                dst,arrival = destination(edge,lane,release+max(1.,tau[edge]))
                add((dst,k),arrival,float(np.sum(lots[row]*lm[row])))
        pending = {}
        for j in np.flatnonzero(o.get('pending_prohibitions.edge.observed',[])):
            p = (int(o['pending_prohibitions.edge'][j]),int(o['pending_prohibitions.k'][j]))
            pending[p] = min(pending.get(p,self.T+1),int(o['pending_prohibitions.effective_week'][j]))
        forecast = np.asarray(o['demand_forecast.qty'],float)
        backlog = np.asarray(o.get('backlog.qty',np.zeros(len(self.demands))),float)
        def demand_at(di,h):
            row = forecast[di]
            return max(0.,float(row[h] if h < len(row) else np.mean(row[-min(3,len(row)):])))
        def need(dst,lead):
            duration = min(int(np.ceil(lead))+2,self.T-week+1)
            cutoff = week+duration-1
            if dst not in self.di:
                duration = min(lead+2.,max(0.,self.T-week+1))
                target = self.rate.get(dst,0.)*duration
                return max(0.,target-remaining.get(dst,0.)-sum(q for t,q in projected.get(dst,[]) if t <= week+int(np.ceil(duration))))
            di = self.di[dst]
            inv = remaining.get(dst,0.)
            arrivals = {}
            for t,q in projected.get(dst,[]):
                arrivals[t] = arrivals.get(t,0.)+q
            if self.carry[dst]:
                target = float(backlog[di])+sum(demand_at(di,h) for h in range(duration))
                return max(0.,target-inv-sum(q for t,q in projected.get(dst,[]) if t <= cutoff))
            arrival = week+int(np.ceil(lead))
            for t in range(week,min(arrival,cutoff+1)):
                inv = max(0.,inv+arrivals.get(t,0.)-demand_at(di,t-week))
            target = sum(demand_at(di,t-week) for t in range(arrival,cutoff+1))
            return max(0.,target-inv-sum(q for t,q in projected.get(dst,[]) if arrival <= t <= cutoff))
        edge_left = np.maximum(0.,u.copy())
        cp_left = {}
        for pool in ('tb','ct'):
            caps = self.field(o,'graph_now.kappa.'+pool,np.full(len(self.cp),np.inf),True)
            for cp,ci in self.cp.items():
                cp_left[(cp,pool)] = max(0.,float(caps[ci])*float(opened[ci])) if opened[ci] > 0 else 0.
        candidates = []
        for i,k,lane,es,cps,src,dst in self.routes:
            if not mask[i] or any(banned[e,k] for e in es):
                continue
            lead = sum(max(1.,tau[e]) for e in es)
            if week+lead > self.T+1 or min(u[e] for e in es) <= 0:
                continue
            elapsed = 0.
            risk = False
            for e in es:
                if pending.get((e,k),self.T+1) <= week+elapsed:
                    risk = True
                elapsed += max(1.,tau[e])
            if risk or any(opened[self.cp[cp]] <= .001 for cp in cps if cp in self.cp):
                continue
            cost = sum(c[e]+tariff[e,k]*self.v[k] for e in es)
            priority = max(1.,self.priority.get(dst,self.maxpi*.25))
            score = (cost+.002*self.v[k]*lead)/priority
            candidates.append((self.depth.get(dst,0),score,lead,i))
        flows = np.zeros(len(self.routes))
        for _,score,lead,i in sorted(candidates):
            _,k,lane,es,cps,src,dst = self.routes[i]
            q = min(need(dst,lead),available.get(src,0.),min(edge_left[e] for e in es))
            for cp in cps:
                if cp in self.cp:
                    q = min(q,cp_left[(cp,self.pool[k])])
            if not np.isfinite(q) or q <= 0:
                continue
            flows[i] = q
            available[src] = max(0.,available.get(src,0.)-q)
            remaining[src] = max(0.,remaining.get(src,0.)-q)
            for e in es:
                edge_left[e] = max(0.,edge_left[e]-q)
            for cp in cps:
                if cp in self.cp:
                    cp_left[(cp,self.pool[k])] = max(0.,cp_left[(cp,self.pool[k])]-q)
            add(dst,week+int(np.ceil(lead)),q)
        return {'flows':flows}
