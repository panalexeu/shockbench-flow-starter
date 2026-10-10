# 0.6050840578424073
import numpy as np

class Agent:
    def __init__(self, config):
        self.s = s = config['static']
        self.l = config['layout']
        self.T = int(config['T'])
        self.tail = np.asarray(s['edges']['tail'], int)
        self.head = np.asarray(s['edges']['head'], int)
        self.u0 = np.array([0 if x is None else x for x in s['edges']['u0']], float)
        self.c0 = np.asarray(s['edges']['c0'], float)
        self.t0 = np.asarray(s['edges']['tau0'], float)
        self.v = np.asarray(s['commodities']['v'], float)
        self.cp = {n:i for i,n in enumerate(self.l['chokepoints'])}
        self.slots = []
        self.groups = {}
        for i,(e,k,lane) in enumerate(zip(s['action_slots']['edge'], s['action_slots']['k'], s['action_slots']['lane'])):
            path = list(s['lanes']['edges'][lane]) if lane is not None and lane >= 0 else [e]
            a,b = int(self.tail[path[0]]), int(self.head[path[-1]])
            self.slots.append((a,b,k,path,lane))
            self.groups.setdefault((b,k), []).append(i)
        self.stock_idx = {tuple(x):i for i,x in enumerate(self.l['stock_slots'])}
        self.sink_idx = {tuple(x):i for i,x in enumerate(self.l['demands'])}
        names = [{v:i for i,v in enumerate(s[t]['id'])} for t in ['edges','commodities']]
        pipe = s.get('instance',{}).get('initial_state',{}).get('pipeline',[])
        if isinstance(pipe,dict):
            pipe = [dict(zip(pipe,x)) for x in zip(*pipe.values())] if isinstance(pipe.get('edge'),list) else list(pipe.values())
        totals = {}
        for p in pipe:
            try:
                e = p['edge']
                k = p.get('k',p.get('commodity'))
                e = int(names[0].get(e,e))
                k = int(names[1].get(k,k))
                totals[e,k] = totals.get((e,k),0) + float(p.get('qty',p.get('quantity',0)))
            except (TypeError,ValueError,KeyError):
                pass
        self.rates = {}
        for (e,k),q in totals.items():
            key = (int(self.head[e]),k)
            self.rates[key] = self.rates.get(key,0) + q/max(1,self.t0[e])
        for key,ids in self.groups.items():
            if key not in self.rates:
                r = sum(totals.get((self.slots[i][3][0],key[1]),0)/max(1,self.t0[self.slots[i][3][0]]) for i in ids)
                self.rates[key] = r if r > 0 else .55*max(min(self.u0[e] for e in self.slots[i][3]) for i in ids)
        self.downstream = {key:0. for key in self.sink_idx}
        for _ in range(len(self.tail)):
            changed = False
            for a,b,k,path,lane in self.slots:
                if (b,k) in self.downstream:
                    d = self.downstream[b,k] + sum(self.t0[e] for e in path)
                    if d < self.downstream.get((a,k),float('inf')):
                        self.downstream[a,k] = d
                        changed = True
            if not changed:
                break
        self.base = None
        self.last = {}
        self.cache = {}

    def act(self,o):
        def read(key,default):
            fallback = self.cache.get(key,default) if key != 'stock.qty' else default
            x = np.asarray(o.get(key,fallback))
            m = o.get(key+'.observed')
            if m is not None and np.shape(m) == x.shape:
                x = np.where(np.asarray(m)>0,x,fallback)
            self.cache[key] = x.copy()
            return x
        week = int(o['week'][0])
        remain = self.T-week+1
        u = read('graph_now.u',self.u0).astype(float)
        c = read('graph_now.c',self.c0)
        tau = read('graph_now.tau',self.t0)
        tariff = read('graph_now.tariff',np.zeros((len(u),len(self.v))))
        opening = read('graph_now.open',np.ones(len(self.cp)))
        mask = np.asarray(o.get('action_mask',np.ones(len(self.slots))))
        vals = read('stock.qty',np.array([self.last.get(tuple(x),0) for x in self.l['stock_slots']]))
        stock = {key:max(0,float(vals[j])) for key,j in self.stock_idx.items()}
        avail = dict(stock)
        supply = read('graph_now.supply.avail',np.zeros(len(self.l['supply_slots'])))
        for j,key in enumerate(self.l['supply_slots']):
            key = tuple(key)
            avail[key] = avail.get(key,0) + max(0,float(supply[j]))
        forecast = read('demand_forecast.qty',np.zeros_like(o['demand_forecast.qty']))
        mean = np.mean(forecast,axis=1)
        if self.base is None:
            self.base = np.maximum(mean,1e-8)
        ratios = {}
        for j,(_,k) in enumerate(self.l['demands']):
            ratios.setdefault(k,[]).append(mean[j]/self.base[j])
        rates = {key:r*float(np.clip(np.mean(ratios.get(key[1],[1.])),.5,1.8)) for key,r in self.rates.items()}
        for key,j in self.sink_idx.items():
            rates[key] = float(mean[j])
        ends = {}
        for j in np.flatnonzero(o.get('closure_end.end_week.observed',[])):
            n = int(o['closure_end.chokepoint'][j])
            ends[n] = max(ends.get(n,week),int(o['closure_end.end_week'][j]))
        pending = {}
        for j in np.flatnonzero(o.get('pending_prohibitions.edge.observed',[])):
            key = (int(o['pending_prohibitions.edge'][j]),int(o['pending_prohibitions.k'][j]))
            pending[key] = min(pending.get(key,self.T+1),int(o['pending_prohibitions.effective_week'][j]))
        prohibited = read('graph_now.prohibited',np.zeros((len(u),len(self.v))))
        warning = read('warning.score',np.zeros(len(self.l['warning_units'])))
        cpwarning = {}
        regwarning = {}
        for j,(kind,ident) in enumerate(self.l['warning_units']):
            value = float(np.clip(warning[j],0,1))
            if kind == 'chokepoint':
                cpwarning[ident] = value
            elif kind == 'region':
                regwarning[ident] = value
        pools = self.s['commodities']['pool']
        queued = {}
        lot_keys = self.l.get('lot_keys',[])
        lots = np.asarray(o.get('queue_lots.qty',np.zeros((len(lot_keys),self.T))))
        for j,(n,k,lane,e) in enumerate(lot_keys):
            key = (n,pools[k])
            queued[key] = queued.get(key,0.) + float(np.sum(lots[j]))
        throughput = {}
        for pool in set(pools):
            key = 'graph_now.kappa.'+str(pool)
            if key in o:
                throughput[pool] = read(key,np.asarray(o[key]))
        def congestion(n,k):
            pool = pools[k]
            if pool not in throughput or n not in self.cp:
                return 0.
            cap = float(throughput[pool][self.cp[n]])*max(.2,float(opening[self.cp[n]]))
            return min(6.,max(0.,queued.get((n,pool),0.)/max(cap,1e-8)-1.)) if cap>0 else 0.
        def journey(path,k,start):
            time = float(start)
            rel = 1.
            for e in path:
                if prohibited[e,k] or pending.get((e,k),self.T+1)<=time:
                    rel *= .05
                time += max(0,float(tau[e]))
                n = int(self.head[e])
                if n in self.cp:
                    op = float(opening[self.cp[n]])
                    if op<.99:
                        time += max(0,ends.get(n,week+5)-time)*(1-op)
                        if n not in ends:
                            rel *= .65+.35*op
                    time += congestion(n,k)*max(0.,1.-(time-week)/8.)
            return time,rel
        incoming = {}
        def add(key,q,time,rel=1.):
            if time<=self.T and q>0:
                incoming.setdefault(key,[]).append((q,time,rel))
        for j in np.flatnonzero(o.get('pipeline.qty.observed',[])):
            e,k = int(o['pipeline.edge'][j]),int(o['pipeline.k'][j])
            lane = int(o['pipeline.lane'][j])
            if not o.get('pipeline.lane.observed',np.ones(len(o['pipeline.qty'])))[j]:
                lane = -1
            path = list(self.s['lanes']['edges'][lane]) if 0<=lane<len(self.s['lanes']['edges']) else [e]
            rest = path[path.index(e)+1:] if e in path else []
            arrival = float(o['pipeline.arrival_week'][j])
            n = int(self.head[e])
            if n in self.cp:
                arrival += max(0,ends.get(n,week+5)-arrival)*(1-float(opening[self.cp[n]]))
                arrival += .5*congestion(n,k)
            time,rel = journey(rest,k,arrival)
            add((int(self.head[path[-1]]),k),float(o['pipeline.qty'][j]),time,rel)
        for j,(n,k,lane,e) in enumerate(lot_keys):
            path = list(self.s['lanes']['edges'][lane]) if lane is not None and lane>=0 else [e]
            path = path[path.index(e):] if e in path else [e]
            op = float(opening[self.cp[n]]) if n in self.cp else 1.
            start = week + max(0,ends.get(n,week+5)-week)*(1-op) + .5*congestion(n,k)
            time,rel = journey(path,k,start)
            if op<.99 and n not in ends:
                rel *= .35+.65*op
            add((int(self.head[path[-1]]),k),float(np.sum(lots[j])),time,rel)
        for j in np.flatnonzero(o.get('wip.qty.observed',[])):
            add((int(o['wip.node'][j]),int(o['wip.k'][j])),float(o['wip.qty'][j]),float(o['wip.out_week'][j]))
        routes = {}
        for i,(a,b,k,path,lane) in enumerate(self.slots):
            if not mask[i]:
                continue
            time,rel = journey(path,k,week)
            lead = max(1.,time-week)
            cap = max(0.,min(float(u[e]) for e in path))
            freight = sum(float(c[e])+float(tariff[e,k])*self.v[k] for e in path)
            score = freight/max(1,self.v[k]) + .035*lead + 10*(1-rel)
            if cap>0 and time+self.downstream.get((b,k),0)<=self.T:
                risk = max([cpwarning.get(int(self.head[e]),0) for e in path]+[regwarning.get(self.s['nodes']['region'][a],0),regwarning.get(self.s['nodes']['region'][b],0)])
                routes[i] = (lead,cap,score,risk,rel,freight)
        backlog = read('backlog.qty',np.zeros(len(mean)))
        candidates = []
        deficits = {}
        for key,ids in self.groups.items():
            valid = [i for i in ids if i in routes]
            if not valid:
                continue
            stocked = [i for i in valid if avail.get((self.slots[i][0],key[1]),0)>0]
            best = min(stocked or valid,key=lambda i:routes[i][2])
            bestlead = routes[best][0]
            buffer = (17.0 if key in self.sink_idx else 12.5) + 4.0*max(0.,routes[best][3]-.45)
            deadline = min(pending.get((e,key[1]),self.T+1) for e in self.slots[best][3])
            extra = 4. if routes[best][4]>.95 and week<deadline<=week+bestlead+3 else 0.
            usable = max(0.,remain-self.downstream.get(key,0))
            horizon = min(bestlead+buffer+extra,usable)
            target = rates.get(key,0)*horizon
            if key in self.sink_idx:
                j = self.sink_idx[key]
                h = min(int(np.floor(horizon)),forecast.shape[1])
                target = float(np.sum(forecast[j,:h]))
                if h<forecast.shape[1]:
                    target += (horizon-h)*float(forecast[j,h])
                else:
                    target += max(0,horizon-h)*rates[key]
                target += float(backlog[j])
            position = stock.get(key,0)
            final = horizon>=usable-1e-8
            for q,time,r in incoming.get(key,[]):
                if final:
                    weight = float(time<=min(self.T,week+usable))
                else:
                    weight = min(1.,max(0.,(week+horizon+1-time)/max(1.,buffer)))
                position += q*r*weight
            deficit = max(0,target-position)
            rate = max(1.,rates.get(key,0))
            for i in valid:
                lead,cap,score,risk,rel,freight = routes[i]
                deficits[i] = deficit
                urgency = deficit/rate-.65*max(0,lead-bestlead)-3.5*(1-rel)-1.4*score
                if lead<bestlead:
                    balance = stock.get(key,0)
                    if key in self.sink_idx:
                        balance -= float(backlog[self.sink_idx[key]])
                    balance += sum(q*r for q,time,r in incoming.get(key,[]) if time<=week)
                    shortage = 0.
                    for dt in range(1,min(remain,int(np.ceil(bestlead)))+1):
                        usage = rate
                        if key in self.sink_idx and dt<=forecast.shape[1]:
                            usage = float(forecast[self.sink_idx[key],dt-1])
                        balance -= usage
                        for q,time,r in incoming.get(key,[]):
                            if week+dt-1<time<=week+dt:
                                balance += q*r
                        if dt>=lead:
                            shortage = max(shortage,-balance/rate)
                    shortage = max(0.,shortage)
                    urgency += 1.4*min(bestlead-lead,shortage)
                    if freight>routes[best][5]*1.5+1:
                        deficits[i] = min(deficit,rate*(shortage+.75))
                        if shortage<=0:
                            urgency -= 3.
                candidates.append((-urgency,score,i,key))
        flows = np.zeros(len(self.slots))
        sent = {}
        edge_left = np.maximum(u.copy(),0.)
        for _,_,i,key in sorted(candidates):
            a,b,k,path,lane = self.slots[i]
            source = (a,k)
            need = max(0,deficits[i]-sent.get(key,0))
            q = min(need,routes[i][1],avail.get(source,0.),edge_left[path[0]])
            if q>0 and np.isfinite(q):
                flows[i] = q
                sent[key] = sent.get(key,0)+q
                avail[source] -= q
                edge_left[path[0]] -= q
        self.last = {key:max(0,avail.get(key,0)) for key in stock}
        return {'flows':flows}
