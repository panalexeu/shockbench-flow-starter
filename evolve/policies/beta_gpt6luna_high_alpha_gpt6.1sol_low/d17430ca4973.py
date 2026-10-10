# 0.5384263474491618
import numpy as np

class Agent:
    def __init__(self, config):
        self.s = s = config['static']
        self.l = l = config['layout']
        self.T = int(config['T'])
        e = s['edges']
        self.tail = np.array(e['tail'], int)
        self.head = np.array(e['head'], int)
        self.u0 = np.array([0 if x is None else x for x in e['u0']], float)
        self.c0 = np.array([0 if x is None else x for x in e['c0']], float)
        self.t0 = np.array([1 if x is None else x for x in e['tau0']], float)
        self.v = np.array(s['commodities']['v'], float)
        self.pool = s['commodities']['pool']
        self.stock = [tuple(map(int, p)) for p in l['stock_slots']]
        self.supply = [tuple(map(int, p)) for p in l['supply_slots']]
        self.demands = [tuple(map(int, p)) for p in l['demands']]
        self.di = {p:i for i,p in enumerate(self.demands)}
        self.cp = {int(n):i for i,n in enumerate(l['chokepoints'])}
        self.pi = np.array(s['sinks']['pi'], float)
        self.maxpi = max(1., float(self.pi.max()) if self.pi.size else 1.)
        self.routes = []
        grouped = {}
        slots = s['action_slots']
        for i,(edge,k,lane) in enumerate(zip(slots['edge'],slots['k'],slots['lane'])):
            edge,k = int(edge),int(k)
            lane = -1 if lane is None else int(lane)
            es = [edge] if lane < 0 else list(map(int,s['lanes']['edges'][lane]))
            if not es: continue
            cps = [] if lane < 0 else list(map(int,s['lanes']['chokepoints'][lane]))
            src,dst = int(self.tail[es[0]]),int(self.head[es[-1]])
            self.routes.append((i,k,es,cps,src,dst))
            alt = e['alt_of'][edge] if lane < 0 else s['lanes']['alt_of'][lane]
            grouped.setdefault((src,k,dst),[]).append((alt,min(self.u0[x] for x in es)))
        inc,out = {},{}
        for (src,k,dst),choices in grouped.items():
            base = [q for alt,q in choices if alt is None]
            cap = sum(base) if base else max(q for _,q in choices)
            inc[dst,k] = inc.get((dst,k),0)+cap
            out[src,k] = out.get((src,k),0)+cap
        self.rate = {}
        for p in set(inc)|set(out):
            a,b = inc.get(p),out.get(p)
            self.rate[p] = min(a,b) if a is not None and b is not None else max(a or 0,b or 0)
        reverse = {}
        for _,k,_,_,src,dst in self.routes:
            reverse.setdefault((dst,k),set()).add(src)
        self.priority = {}
        for di,(sink,k) in enumerate(self.demands):
            queue,seen = [(sink,0)],{sink}
            for node,dist in queue:
                p = (node,k)
                self.priority[p] = max(self.priority.get(p,0),self.pi[di]/(1+.12*dist))
                for src in reverse.get(p,()):
                    if src not in seen:
                        seen.add(src)
                        queue.append((src,dist+1))

    def field(self,o,key,fb):
        x = np.asarray(o.get(key,fb))
        m = o.get(key+'.observed')
        return np.where(m,x,fb) if m is not None and np.shape(m)==x.shape else x.copy()

    def act(self,o):
        w = int(o['week'][0])
        ne,nk = len(self.tail),len(self.v)
        ns = len(self.s['action_slots']['edge'])
        u = self.field(o,'graph_now.u',self.u0).astype(float)
        c = self.field(o,'graph_now.c',self.c0)
        tau = np.maximum(1,self.field(o,'graph_now.tau',self.t0))
        tar = self.field(o,'graph_now.tariff',np.zeros((ne,nk)))
        ban = self.field(o,'graph_now.prohibited',np.zeros((ne,nk))).astype(bool)
        opened = self.field(o,'graph_now.open',np.ones(len(self.cp)))
        mask = np.asarray(o.get('action_mask',np.ones(ns)))
        left = {}
        for pool in ('tb','ct'):
            caps = self.field(o,'graph_now.kappa.'+pool,np.full(len(self.cp),np.inf))
            for cp,ci in self.cp.items():
                left[cp,pool] = float(caps[ci])*float(opened[ci]) if opened[ci]>0 else 0.
        available,stocks,proj = {},{},{}
        for p,q in zip(self.stock,self.field(o,'stock.qty',np.zeros(len(self.stock)))):
            available[p] = stocks[p] = max(0.,float(q))
        for p,q in zip(self.supply,self.field(o,'graph_now.supply.avail',np.zeros(len(self.supply)))):
            available[p] = available.get(p,0)+max(0.,float(q))
        def add(p,t,q):
            if q>0: proj.setdefault(p,[]).append((t,float(q)))
        lanes = self.s['lanes']['edges']
        def remainder(edge,lane):
            es = list(map(int,lanes[lane])) if 0<=lane<len(lanes) else [edge]
            if edge not in es: es=[edge]
            es = es[es.index(edge):]
            return int(self.head[es[-1]]),es
        pq = np.asarray(o.get('pipeline.qty',[]))
        pm = o.get('pipeline.qty.observed',np.ones(pq.size))
        for j,q in enumerate(pq):
            if not pm[j] or q<=0: continue
            edge = int(o['pipeline.edge'][j])
            if not 0<=edge<ne: continue
            lm = o.get('pipeline.lane.observed',np.zeros(pq.size))
            lane = int(o['pipeline.lane'][j]) if lm[j] else -1
            dst,es = remainder(edge,lane)
            t = int(o['pipeline.arrival_week'][j])+int(sum(tau[e] for e in es[1:]))
            add((dst,int(o['pipeline.k'][j])),max(w,t),q)
        wq = np.asarray(o.get('wip.qty',[]))
        wm = o.get('wip.qty.observed',np.ones(wq.size))
        for j,q in enumerate(wq):
            if wm[j] and q>0:
                add((int(o['wip.node'][j]),int(o['wip.k'][j])),max(w,int(o['wip.out_week'][j])),q)
        ends = {}
        ce = o.get('closure_end.chokepoint',[])
        cem = o.get('closure_end.chokepoint.observed',np.zeros(len(ce)))
        cwm = o.get('closure_end.end_week.observed',np.zeros(len(ce)))
        for j,cp in enumerate(ce):
            if cem[j] and cwm[j]: ends[int(cp)] = max(ends.get(int(cp),w),int(o['closure_end.end_week'][j])+1)
        lots = np.asarray(o.get('queue_lots.qty',np.zeros((0,0))))
        live = o.get('queue_lots.qty.observed',np.ones(lots.shape))
        for row,key in enumerate(self.l.get('lot_keys',[])[:lots.shape[0]]):
            cp,k,lane,edge = key
            edge = int(edge)
            if not 0<=edge<ne: continue
            dst,es = remainder(edge,-1 if lane is None else int(lane))
            q = float(np.sum(np.where(live[row],lots[row],0)))
            start = max(w,ends.get(int(cp),w))
            ci = self.cp.get(int(cp))
            if ci is not None and opened[ci]<=0 and int(cp) not in ends: start += 3
            add((dst,int(k)),start+int(sum(tau[e] for e in es)),q)
        pending = {}
        pe = o.get('pending_prohibitions.edge',[])
        pm = o.get('pending_prohibitions.edge.observed',np.zeros(len(pe)))
        for j,e in enumerate(pe):
            if pm[j]:
                p = (int(e),int(o['pending_prohibitions.k'][j]))
                pending[p] = min(pending.get(p,self.T+1),int(o['pending_prohibitions.effective_week'][j]))
        forecast = np.asarray(o.get('demand_forecast.qty',np.zeros((len(self.demands),1))),float)
        backlog = np.asarray(o.get('backlog.qty',np.zeros(len(self.demands))))
        def target(p,lead):
            if p in self.di:
                di = self.di[p]
                h = min(self.T-w+1,int(np.ceil(lead))+2)
                row = forecast[di]
                n = min(h,len(row))
                total = float(backlog[di])+float(row[:n].sum())
                if h>n and n: total += (h-n)*float(np.mean(row[-min(3,n):]))
                return total,w+h-1
            duration = min(lead+2.,max(1,self.T-w+1))
            return self.rate.get(p,0)*duration,w+int(np.ceil(duration))
        def coverage(p,cut):
            return stocks.get(p,0)+sum(q for t,q in proj.get(p,()) if t<=cut)
        candidates = []
        for r in self.routes:
            i,k,es,cps,src,dst = r
            if not mask[i] or any(ban[e,k] for e in es): continue
            lead = float(sum(tau[e] for e in es))
            if w+lead>self.T+1: continue
            elapsed = 0
            risky = False
            for e in es:
                if pending.get((e,k),self.T+1)<=w+elapsed: risky=True
                elapsed += tau[e]
            if risky: continue
            cap = min(u[e] for e in es)
            for cp in cps: cap=min(cap,left.get((cp,self.pool[k]),np.inf))
            if cap<=0: continue
            p = dst,k
            total,cut = target(p,lead)
            have = coverage(p,cut)
            if total<=have: continue
            priority = max(1.,self.priority.get(p,self.maxpi*.25))
            cost = sum(c[e]+tar[e,k]*self.v[k] for e in es)
            urgency = 1.+.25*max(0.,1.-have/max(total,1e-9))
            score = (cost+.002*self.v[k]*lead)/(priority*urgency)
            candidates.append((score,lead,i,cap,r))
        flows = np.zeros(ns)
        candidates.sort(key=lambda x:(x[0],x[1],x[2]))
        for _,lead,i,cap,r in candidates:
            _,k,es,cps,src,dst = r
            p,ps = (dst,k),(src,k)
            total,cut = target(p,lead)
            q = min(cap,max(0,total-coverage(p,cut)),available.get(ps,0))
            for e in es: q=min(q,max(0,u[e]))
            for cp in cps: q=min(q,left.get((cp,self.pool[k]),np.inf))
            if q<=0 or not np.isfinite(q): continue
            flows[i] = q
            available[ps] -= q
            stocks[ps] = max(0,stocks.get(ps,0)-q)
            for e in es: u[e]=max(0,u[e]-q)
            for cp in cps:
                key=cp,self.pool[k]
                if key in left: left[key]=max(0,left[key]-q)
            add(p,w+int(np.ceil(lead)),q)
        return {'flows':flows}
