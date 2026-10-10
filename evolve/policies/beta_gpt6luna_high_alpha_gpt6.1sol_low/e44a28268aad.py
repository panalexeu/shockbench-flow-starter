# 0.4959246178916349
import numpy as np

class Agent:
    def __init__(self, config):
        self.s = s = config['static']
        self.l = l = config['layout']
        self.T = int(config['T'])
        self.memory = {}
        e = s['edges']
        self.tail = np.asarray(e['tail'], int)
        self.head = np.asarray(e['head'], int)
        self.u0 = np.asarray([0 if x is None else x for x in e['u0']], float)
        self.c0 = np.asarray([0 if x is None else x for x in e['c0']], float)
        self.tau0 = np.asarray([1 if x is None else x for x in e['tau0']], float)
        self.v = np.asarray(s['commodities']['v'], float)
        self.pool = s['commodities']['pool']
        self.stocks = [tuple(p) for p in l['stock_slots']]
        self.supplies = [tuple(p) for p in l['supply_slots']]
        self.demands = [tuple(p) for p in l['demands']]
        self.di = {p:i for i,p in enumerate(self.demands)}
        self.cp = {n:i for i,n in enumerate(l['chokepoints'])}
        sinkpi = {(n,k):float(pi) for n,k,pi in zip(s['sinks']['node'],s['sinks']['k'],s['sinks']['pi'])}
        self.pi = np.asarray([sinkpi.get(p,1) for p in self.demands])
        self.maxpi = max(1., float(np.max(self.pi)) if self.pi.size else 1.)
        self.routes = []
        grouped = {}
        a = s['action_slots']
        for i,(edge,k,lane) in enumerate(zip(a['edge'],a['k'],a['lane'])):
            lane = -1 if lane is None else int(lane)
            es = [int(edge)] if lane < 0 else list(s['lanes']['edges'][lane])
            cps = [] if lane < 0 else list(s['lanes']['chokepoints'][lane])
            src,dst = int(self.tail[es[0]]),int(self.head[es[-1]])
            r = (i,int(k),es,cps,src,dst)
            self.routes.append(r)
            alt = e['alt_of'][edge] if lane < 0 else s['lanes']['alt_of'][lane]
            grouped.setdefault((src,k,dst),[]).append((alt,min(self.u0[x] for x in es)))
        incoming,outgoing = {},{}
        for (src,k,dst), choices in grouped.items():
            base = [cap for alt,cap in choices if alt is None]
            cap = sum(base) if base else max(cap for _,cap in choices)
            incoming[dst,k] = incoming.get((dst,k),0)+cap
            outgoing[src,k] = outgoing.get((src,k),0)+cap
        self.rate = {}
        for p in set(incoming)|set(outgoing):
            x,y = incoming.get(p),outgoing.get(p)
            self.rate[p] = min(x,y) if x is not None and y is not None else max(x or 0,y or 0)
        reverse = {}
        for _,k,es,cps,src,dst in self.routes:
            reverse.setdefault((dst,k),set()).add(src)
        self.priority = {}
        for i,(sink,k) in enumerate(self.demands):
            queue = [(sink,0)]
            seen = {sink}
            for node,d in queue:
                p = (node,k)
                self.priority[p] = max(self.priority.get(p,0),self.pi[i]/(1+.12*d))
                for prev in reverse.get(p,()):
                    if prev not in seen:
                        seen.add(prev)
                        queue.append((prev,d+1))

    def field(self,o,name,default,remember=False):
        fallback = self.memory.get(name,np.asarray(default)) if remember else np.asarray(default)
        if name not in o:
            return np.array(fallback,copy=True)
        x = np.asarray(o[name])
        mask = o.get(name+'.observed')
        if mask is not None and np.shape(mask)==x.shape:
            x = np.where(mask,x,fallback)
        else:
            x = x.copy()
        if remember:
            self.memory[name] = x.copy()
        return x

    def act(self,o):
        week = int(o['week'][0])
        ne,nk = len(self.tail),len(self.v)
        u = self.field(o,'graph_now.u',self.u0,True)
        c = self.field(o,'graph_now.c',self.c0,True)
        tau = self.field(o,'graph_now.tau',self.tau0,True)
        tariff = self.field(o,'graph_now.tariff',np.zeros((ne,nk)),True)
        prohibited = self.field(o,'graph_now.prohibited',np.zeros((ne,nk)),True)
        opened = self.field(o,'graph_now.open',np.ones(len(self.cp)),True)
        mask = np.asarray(o.get('action_mask',np.ones(len(self.routes))))
        available,stock,projected = {},{},{}
        for p,q in zip(self.stocks,self.field(o,'stock.qty',np.zeros(len(self.stocks)))):
            available[p] = stock[p] = max(0.,float(q))
        for p,q in zip(self.supplies,self.field(o,'graph_now.supply.avail',np.zeros(len(self.supplies)))):
            available[p] = available.get(p,0)+max(0.,float(q))
        closures = {}
        for j in np.flatnonzero(o.get('closure_end.chokepoint.observed',[])):
            cp = int(o['closure_end.chokepoint'][j])
            if o.get('closure_end.end_week.observed',np.zeros_like(o['closure_end.end_week']))[j]:
                closures[cp] = max(closures.get(cp,week),int(o['closure_end.end_week'][j]))
        def final(edge,lane,arrival):
            dst = int(self.head[edge])
            if lane is not None and 0 <= int(lane) < len(self.s['lanes']['edges']):
                es = self.s['lanes']['edges'][int(lane)]
                dst = int(self.head[es[-1]])
                if edge in es:
                    for x in es[es.index(edge)+1:]:
                        cp = int(self.tail[x])
                        if cp in self.cp and opened[self.cp[cp]] <= .001:
                            arrival = max(arrival,closures.get(cp,week+4))
                        arrival += max(1.,float(tau[x]))
            return dst,int(np.ceil(arrival))
        qty = np.asarray(o.get('pipeline.qty',[]))
        for j in np.flatnonzero(o.get('pipeline.qty.observed',np.ones(qty.size))):
            q = float(qty[j])
            edge = int(o['pipeline.edge'][j])
            if q <= 0 or not 0 <= edge < ne:
                continue
            lane = int(o['pipeline.lane'][j]) if o.get('pipeline.lane.observed',np.ones(qty.size))[j] else -1
            dst,arrival = final(edge,lane,max(week,int(o['pipeline.arrival_week'][j])))
            projected.setdefault((dst,int(o['pipeline.k'][j])),[]).append((arrival,q))
        for j in np.flatnonzero(o.get('wip.qty.observed',[])):
            p = (int(o['wip.node'][j]),int(o['wip.k'][j]))
            projected.setdefault(p,[]).append((int(o['wip.out_week'][j]),max(0.,float(o['wip.qty'][j]))))
        lots = np.asarray(o.get('queue_lots.qty',[]))
        if lots.ndim == 2:
            live = np.asarray(o.get('queue_lots.qty.observed',np.ones_like(lots)))
            for row,(cp,k,lane,edge) in enumerate(self.l.get('lot_keys',[])[:lots.shape[0]]):
                q = float(np.sum(np.where(live[row],lots[row],0)))
                if q <= 0:
                    continue
                release = week
                if cp in self.cp and opened[self.cp[cp]] <= .001:
                    release = closures.get(cp,week+4)
                dst,arrival = final(int(edge),lane,release+max(1.,float(tau[edge])))
                projected.setdefault((dst,int(k)),[]).append((arrival,q))
        pending = {}
        for j in np.flatnonzero(o.get('pending_prohibitions.edge.observed',[])):
            p = (int(o['pending_prohibitions.edge'][j]),int(o['pending_prohibitions.k'][j]))
            pending[p] = min(pending.get(p,self.T+1),int(o['pending_prohibitions.effective_week'][j]))
        warnings = {}
        scores = self.field(o,'warning.score',np.zeros(len(self.l['warning_units'])))
        for key,score in zip(self.l['warning_units'],scores):
            warnings[tuple(key)] = float(np.clip(score,0,1))
        forecast = np.asarray(o.get('demand_forecast.qty',np.zeros((len(self.demands),1))))
        backlog = np.asarray(o.get('backlog.qty',np.zeros(len(self.demands))))
        def need(r,lead,buffer):
            _,k,es,cps,src,dst = r
            p = (dst,k)
            duration = min(lead+buffer,max(0,self.T-week+1))
            if p in self.di:
                d = self.di[p]
                row = np.atleast_1d(forecast[d])
                h = max(1,int(np.ceil(duration)))
                target = float(backlog[d])+float(np.sum(row[:h]))
                if h > len(row) and len(row):
                    target += (h-len(row))*float(np.mean(row[-min(3,len(row)):]))
                cutoff = week+h-1
            else:
                target = self.rate.get(p,0)*duration
                cutoff = week+int(np.ceil(duration))
            return max(0.,target-stock.get(p,0)-sum(q for t,q in projected.get(p,()) if t<=cutoff))
        left = np.maximum(0.,u.copy())
        cpleft = {}
        for pool in ('tb','ct'):
            caps = self.field(o,'graph_now.kappa.'+pool,np.full(len(self.cp),np.inf),True)
            for cp,ci in self.cp.items():
                cpleft[cp,pool] = 0. if opened[ci]<=0 else max(0.,float(caps[ci])*float(opened[ci]))
        candidates = []
        for r in self.routes:
            i,k,es,cps,src,dst = r
            if not mask[i] or any(prohibited[e,k] for e in es):
                continue
            lead = sum(max(1.,float(tau[e])) for e in es)
            if week+lead > self.T+1:
                continue
            elapsed,risk = 0.,False
            for e in es:
                if pending.get((e,k),self.T+1)<=week+elapsed:
                    risk = True
                elapsed += max(1.,float(tau[e]))
            if risk:
                continue
            cap = min(max(0.,float(u[e])) for e in es)
            for cp in cps:
                cap = min(cap,cpleft.get((cp,self.pool[k]),np.inf))
            if cap<=0:
                continue
            warning = max([warnings.get(('chokepoint',cp),0.) for cp in cps]+[warnings.get(('region',self.s['nodes']['region'][src]),0.)])
            buffer = 2.+1.5*max(0.,warning-.5)
            requirement = need(r,lead,buffer)
            if requirement<=0:
                continue
            priority = max(1.,self.priority.get((dst,k),self.maxpi*.25))
            cost = sum(float(c[e])+float(tariff[e,k])*self.v[k] for e in es)
            score = (cost+.002*self.v[k]*lead)/priority
            candidates.append((score,lead,i,cap,buffer))
        flows = np.zeros(len(self.s['action_slots']['edge']))
        for _,lead,i,cap,buffer in sorted(candidates):
            r = self.routes[i]
            _,k,es,cps,src,dst = r
            p = (src,k)
            q = min(cap,need(r,lead,buffer),available.get(p,0),min(left[e] for e in es))
            for cp in cps:
                q = min(q,cpleft.get((cp,self.pool[k]),np.inf))
            if q<=0 or not np.isfinite(q):
                continue
            flows[i] = q
            available[p] = max(0.,available.get(p,0)-q)
            stock[p] = max(0.,stock.get(p,0)-q)
            for e in es:
                left[e] = max(0.,left[e]-q)
            for cp in cps:
                key = (cp,self.pool[k])
                if key in cpleft:
                    cpleft[key] = max(0.,cpleft[key]-q)
            projected.setdefault((dst,k),[]).append((week+int(np.ceil(lead)),q))
        return {'flows':flows}
