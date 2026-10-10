# -0.03066853666558373
import numpy as np

class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.T = config['T']
        s = self.s
        self.tail = np.asarray(s['edges']['tail'], int)
        self.head = np.asarray(s['edges']['head'], int)
        self.u0 = np.asarray([0 if v is None else v for v in s['edges']['u0']], float)
        self.c0 = np.asarray(s['edges']['c0'], float)
        self.t0 = np.asarray(s['edges']['tau0'], float)
        self.v = np.asarray(s['commodities']['v'], float)
        self.cp = {n:i for i,n in enumerate(self.l['chokepoints'])}
        self.slots = []
        self.groups = {}
        for i,(e,k,lane) in enumerate(zip(s['action_slots']['edge'],s['action_slots']['k'],s['action_slots']['lane'])):
            path = list(s['lanes']['edges'][lane]) if lane is not None and lane >= 0 else [e]
            a,b = int(self.tail[path[0]]),int(self.head[path[-1]])
            self.slots.append((a,b,k,path,lane))
            self.groups.setdefault((b,k),[]).append(i)
        self.stock_idx = {tuple(x):i for i,x in enumerate(self.l['stock_slots'])}
        self.sink_idx = {tuple(x):i for i,x in enumerate(self.l['demands'])}
        self.rates = {}
        names = [{v:i for i,v in enumerate(s[t]['id'])} for t in ['edges','commodities']]
        pipe = s.get('instance',{}).get('initial_state',{}).get('pipeline',[])
        if isinstance(pipe,dict):
            if isinstance(pipe.get('edge'),list):
                pipe = [dict(zip(pipe,x)) for x in zip(*pipe.values())]
            else:
                pipe = list(pipe.values())
        totals = {}
        for p in pipe:
            try:
                e = p['edge']; k = p.get('k',p.get('commodity'))
                e = int(names[0].get(e,e)); k = int(names[1].get(k,k))
                q = float(p.get('qty',p.get('quantity',0)))
                totals[e,k] = totals.get((e,k),0)+q
            except (TypeError,ValueError,KeyError):
                pass
        for (e,k),q in totals.items():
            key = (int(self.head[e]),k)
            self.rates[key] = self.rates.get(key,0)+q/max(1,self.t0[e])
        for key,ids in self.groups.items():
            if key not in self.rates:
                r = sum(totals.get((self.slots[i][3][0],key[1]),0)/max(1,self.t0[self.slots[i][3][0]]) for i in ids)
                self.rates[key] = r if r>0 else .5*max(min(self.u0[e] for e in self.slots[i][3]) for i in ids)
        self.penalty = {}
        for n,k,p in zip(s['sinks']['node'],s['sinks']['k'],s['sinks']['pi']):
            self.penalty[n,k] = float(p)
        self.base = None
        self.last = {}

    def act(self,o):
        def read(key,default):
            x = np.asarray(o.get(key,default))
            m = o.get(key+'.observed')
            return np.where(np.asarray(m)>0,x,default) if m is not None and np.shape(m)==x.shape else x
        week = int(o['week'][0])
        remain = self.T-week+1
        u = read('graph_now.u',self.u0).astype(float)
        c = read('graph_now.c',self.c0)
        tau = read('graph_now.tau',self.t0)
        tariff = read('graph_now.tariff',np.zeros((len(u),len(self.v))))
        opening = read('graph_now.open',np.ones(len(self.cp)))
        mask = np.asarray(o.get('action_mask',np.ones(len(self.slots))))
        stock = {key:max(0,float(read('stock.qty',np.asarray([self.last.get(tuple(x),0) for x in self.l['stock_slots']]))[j])) for key,j in self.stock_idx.items()}
        avail = dict(stock)
        supply = read('graph_now.supply.avail',np.zeros(len(self.l['supply_slots'])))
        for j,key in enumerate(self.l['supply_slots']):
            key = tuple(key)
            avail[key] = avail.get(key,0)+max(0,float(supply[j]))
        incoming = {}
        for j in np.flatnonzero(o.get('pipeline.qty.observed',[])):
            if int(o['pipeline.arrival_week'][j])>self.T:
                continue
            e,k = int(o['pipeline.edge'][j]),int(o['pipeline.k'][j])
            lane = int(o['pipeline.lane'][j])
            if not o.get('pipeline.lane.observed',np.ones(len(o['pipeline.qty'])))[j]:
                lane = -1
            b = int(self.head[self.s['lanes']['edges'][lane][-1]]) if 0<=lane<len(self.s['lanes']['edges']) else int(self.head[e])
            incoming[b,k] = incoming.get((b,k),0)+float(o['pipeline.qty'][j])
        for j,(n,k,lane,e) in enumerate(self.l.get('lot_keys',[])):
            b = int(self.head[self.s['lanes']['edges'][lane][-1]]) if lane is not None and lane>=0 else int(self.head[e])
            incoming[b,k] = incoming.get((b,k),0)+float(np.sum(o['queue_lots.qty'][j]))
        for j in np.flatnonzero(o.get('wip.qty.observed',[])):
            if int(o['wip.out_week'][j])<=self.T:
                key = (int(o['wip.node'][j]),int(o['wip.k'][j]))
                incoming[key] = incoming.get(key,0)+float(o['wip.qty'][j])
        forecast = read('demand_forecast.qty',np.zeros_like(o['demand_forecast.qty']))
        mean = np.mean(forecast,axis=1)
        if self.base is None:
            self.base = np.maximum(mean,1e-8)
        ratios = {}
        for j,(_,k) in enumerate(self.l['demands']):
            ratios.setdefault(k,[]).append(mean[j]/self.base[j])
        global_ratio = float(np.clip(np.mean(mean/self.base),.65,1.6))
        rates = {key:r*float(np.clip(np.mean(ratios.get(key[1],[global_ratio])),.65,1.6)) for key,r in self.rates.items()}
        for key,j in self.sink_idx.items():
            rates[key] = float(mean[j])
        pending = {}
        for j in np.flatnonzero(o.get('pending_prohibitions.edge.observed',[])):
            key = (int(o['pending_prohibitions.edge'][j]),int(o['pending_prohibitions.k'][j]))
            pending[key] = min(pending.get(key,self.T+1),int(o['pending_prohibitions.effective_week'][j]))
        ends = {}
        for j in np.flatnonzero(o.get('closure_end.end_week.observed',[])):
            n = int(o['closure_end.chokepoint'][j])
            ends[n] = max(ends.get(n,week),int(o['closure_end.end_week'][j]))
        routes = {}
        for i,(a,b,k,path,lane) in enumerate(self.slots):
            if not mask[i]:
                continue
            lead = max(1,float(sum(tau[e] for e in path)))
            cap = max(0,float(u[path[0]]))
            risk = 0
            for e in path:
                n = int(self.head[e])
                if n in self.cp:
                    op = float(opening[self.cp[n]])
                    if op<.99:
                        delay = max(1,ends.get(n,week+5)-week)*(1-op)
                        lead += delay
                        risk += 3*(1-op)
                if pending.get((e,k),self.T+1)<=week+lead:
                    risk += 20
            freight = sum(float(c[e])+float(tariff[e,k])*self.v[k] for e in path)
            if cap>0 and week+lead<=self.T:
                routes[i] = (lead,cap,freight/max(1,self.v[k])+.055*lead+risk)
        candidates = []
        needs = {}
        backlog = read('backlog.qty',np.zeros(len(mean)))
        for key,ids in self.groups.items():
            valid = [i for i in ids if i in routes]
            if not valid:
                continue
            best = min(valid,key=lambda i:routes[i][2])
            lead = routes[best][0]
            rate = rates.get(key,0)
            buffer = 2.8 if key in self.sink_idx else 2.0
            target = rate*min(lead+buffer,remain)
            if key in self.sink_idx:
                target += float(backlog[self.sink_idx[key]])
            position = stock.get(key,0)+incoming.get(key,0)
            deficit = max(0,target-position)
            needs[key] = deficit
            coverage = position/max(1,rate)-lead
            value = self.penalty.get(key,self.v[key[1]])/max(1,self.v[key[1]])
            priority = max(.1,buffer-coverage)*(1+.15*np.log1p(value))
            for i in valid:
                candidates.append((-priority,routes[i][2],i,key))
        flows = np.zeros(len(self.slots))
        edge_left = np.maximum(0,u.copy())
        for _,_,i,key in sorted(candidates):
            a,b,k,path,lane = self.slots[i]
            source = (a,k)
            q = min(needs[key],avail.get(source,0),edge_left[path[0]],routes[i][1])
            if q>0:
                flows[i] = q
                needs[key] -= q
                avail[source] -= q
                edge_left[path[0]] -= q
        self.last = {key:avail.get(key,0) for key in stock}
        return {'flows':flows}
