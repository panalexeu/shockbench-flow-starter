# 0.06508197587237133
import numpy as np

class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.T = config['T']
        s = self.s
        self.tail = np.asarray(s['edges']['tail'], int)
        self.head = np.asarray(s['edges']['head'], int)
        self.c0 = np.asarray(s['edges']['c0'], float)
        self.t0 = np.asarray(s['edges']['tau0'], float)
        self.u0 = np.asarray([0 if x is None else x for x in s['edges']['u0']], float)
        self.slots = []
        self.groups = {}
        self.cp = {n:i for i,n in enumerate(self.l['chokepoints'])}
        for i,(e,k,lane) in enumerate(zip(s['action_slots']['edge'], s['action_slots']['k'], s['action_slots']['lane'])):
            path = list(s['lanes']['edges'][lane]) if lane is not None and lane >= 0 else [e]
            a,b = int(self.tail[path[0]]), int(self.head[path[-1]])
            self.slots.append((a,b,k,path,lane))
            self.groups.setdefault((b,k), []).append(i)
        self.rate = {}
        self.nominal = np.zeros(len(self.slots))
        edge_names = {v:i for i,v in enumerate(s['edges']['id'])}
        good_names = {v:i for i,v in enumerate(s['commodities']['id'])}
        lane_names = {v:i for i,v in enumerate(s['lanes']['id'])}
        def index(v, names):
            return names.get(v, v) if isinstance(v,str) else v
        initial = s.get('instance', {}).get('initial_state', {})
        pipe = initial.get('pipeline', [])
        if isinstance(pipe, dict):
            if 'edge' in pipe and isinstance(pipe['edge'], list):
                pipe = [dict(zip(pipe, vals)) for vals in zip(*pipe.values())]
            else:
                pipe = list(pipe.values())
        totals = {}
        for p in pipe:
            if not isinstance(p,dict):
                continue
            try:
                e = int(index(p.get('edge'), edge_names))
                k = int(index(p.get('k',p.get('commodity')), good_names))
                q = float(p.get('qty',p.get('quantity',0)))
                totals[(e,k)] = totals.get((e,k),0) + q
            except (ValueError,TypeError):
                continue
        for (e,k),q in totals.items():
            r = q / max(1.,self.t0[e])
            dest = (int(self.head[e]),k)
            self.rate[dest] = self.rate.get(dest,0.) + r
            for i,(_,_,kk,path,_) in enumerate(self.slots):
                if path[0] == e and kk == k:
                    self.nominal[i] += r
        for key, inds in self.groups.items():
            if key not in self.rate:
                inferred = sum(self.nominal[i] for i in inds)
                if inferred <= 0:
                    inferred = .55 * max((min(self.u0[e] for e in self.slots[i][3]) for i in inds), default=0)
                self.rate[key] = inferred
        self.stock_lookup = {tuple(v):i for i,v in enumerate(self.l['stock_slots'])}
        self.sink_lookup = {tuple(v):i for i,v in enumerate(self.l['demands'])}
        self.base_forecast = None
        self.last = {}

    def act(self, o):
        def read(key, default):
            x = np.asarray(o.get(key,default))
            m = o.get(key+'.observed')
            if m is not None and np.shape(m) == x.shape:
                x = np.where(np.asarray(m)>0,x,default)
            return x
        week = int(o['week'][0])
        u = read('graph_now.u',self.u0).astype(float)
        cost = read('graph_now.c',self.c0).astype(float)
        tau = read('graph_now.tau',self.t0).astype(float)
        mask = np.asarray(o.get('action_mask',np.ones(len(self.slots))))
        opening = read('graph_now.open',np.ones(len(self.cp)))
        tariff = np.asarray(o.get('graph_now.tariff',np.zeros((len(u),len(self.s['commodities']['id'])))))
        stock = {}
        for key,idx in self.stock_lookup.items():
            val = float(o['stock.qty'][idx])
            observed = o.get('stock.qty.observed')
            if observed is not None and not observed[idx]:
                val = self.last.get(key,val)
            stock[key] = max(0.,val)
        available = dict(stock)
        for idx,key in enumerate(self.l['supply_slots']):
            key = tuple(key)
            available[key] = available.get(key,0.) + max(0.,float(o['graph_now.supply.avail'][idx]))
        incoming = {}
        pm = np.asarray(o.get('pipeline.qty.observed',[]))
        for j in np.flatnonzero(pm):
            e,k = int(o['pipeline.edge'][j]),int(o['pipeline.k'][j])
            q = float(o['pipeline.qty'][j])
            lane = int(o['pipeline.lane'][j])
            lm = o.get('pipeline.lane.observed')
            if lm is not None and not lm[j]:
                lane = -1
            if lane >= 0 and lane < len(self.s['lanes']['edges']):
                b = int(self.head[self.s['lanes']['edges'][lane][-1]])
            else:
                b = int(self.head[e])
            key = (b,k)
            incoming[key] = incoming.get(key,0.) + q
        lots = o.get('queue_lots.qty')
        if lots is not None:
            for j,key in enumerate(self.l.get('lot_keys',[])):
                c,k,lane,e = key
                b = int(self.head[self.s['lanes']['edges'][lane][-1]]) if lane is not None and lane >= 0 else int(self.head[e])
                incoming[(b,k)] = incoming.get((b,k),0.) + float(np.sum(lots[j]))
        wm = np.asarray(o.get('wip.qty.observed',[]))
        for j in np.flatnonzero(wm):
            key = (int(o['wip.node'][j]),int(o['wip.k'][j]))
            incoming[key] = incoming.get(key,0.) + float(o['wip.qty'][j])
        forecast = np.asarray(o['demand_forecast.qty'],float)
        if self.base_forecast is None:
            self.base_forecast = np.maximum(forecast.mean(axis=1),1e-8)
        ratios = {}
        for j,(_,k) in enumerate(self.l['demands']):
            ratios.setdefault(k,[]).append(float(forecast[j].mean()/self.base_forecast[j]))
        rates = {key:r*np.clip(np.mean(ratios.get(key[1],[1.])),.5,1.8) for key,r in self.rate.items()}
        backlog = np.asarray(o.get('backlog.qty',np.zeros(len(forecast))))
        for key,j in self.sink_lookup.items():
            rates[key] = max(0.,float(forecast[j].mean()))
        ends = {}
        em = o.get('closure_end.chokepoint.observed',[])
        for j in np.flatnonzero(em):
            if o.get('closure_end.end_week.observed',np.zeros(len(em)))[j]:
                n = int(o['closure_end.chokepoint'][j])
                ends[n] = max(ends.get(n,week),int(o['closure_end.end_week'][j]))
        pending = {}
        for j in np.flatnonzero(o.get('pending_prohibitions.edge.observed',[])):
            key = (int(o['pending_prohibitions.edge'][j]),int(o['pending_prohibitions.k'][j]))
            pending[key] = min(pending.get(key,self.T+1),int(o['pending_prohibitions.effective_week'][j]))
        candidates = []
        route_data = {}
        for i,(a,b,k,path,lane) in enumerate(self.slots):
            if not mask[i]:
                continue
            lead = max(1.,float(sum(tau[e] for e in path)))
            cap = max(0.,min(float(u[e]) for e in path))
            freight = sum(float(cost[e])+float(tariff[e,k])*float(self.s['commodities']['v'][k]) for e in path)
            risk = 0.
            for e in path:
                n = int(self.head[e])
                if n in self.cp:
                    op = float(opening[self.cp[n]])
                    if op < .99:
                        wait = max(0,ends.get(n,week+4)-week)
                        lead += wait*(1-op)
                        risk += (1-op)*2
                if pending.get((e,k),self.T+1) <= week+lead:
                    risk += 12
            if cap <= 0 or week+lead > self.T+1:
                continue
            score = freight/max(1.,float(self.s['commodities']['v'][k])) + .035*lead + risk
            route_data[i] = (lead,cap,score)
        for key,inds in self.groups.items():
            usable = [i for i in inds if i in route_data]
            if not usable:
                continue
            chosen = min(usable,key=lambda i:route_data[i][2])
            lead = route_data[chosen][0]
            buffer = 1.6 if key in self.sink_lookup else 1.0
            target = rates.get(key,0.)*min(lead+buffer,max(0,self.T-week+1))
            if key in self.sink_lookup:
                target += float(backlog[self.sink_lookup[key]])
            deficit = max(0.,target-stock.get(key,0.)-incoming.get(key,0.))
            urgency = deficit/max(1.,rates.get(key,0.))
            for i in usable:
                candidates.append((-urgency,route_data[i][2],i,key,deficit))
        flows = np.zeros(len(self.slots))
        residual = {}
        edge_left = np.maximum(u.copy(),0.)
        for _,_,i,key,deficit in sorted(candidates):
            residual.setdefault(key,deficit)
            a,b,k,path,lane = self.slots[i]
            source = (a,k)
            q = min(residual[key],route_data[i][1],available.get(source,0.),edge_left[path[0]])
            if q <= 0:
                continue
            flows[i] = q
            residual[key] -= q
            available[source] = available.get(source,0.)-q
            edge_left[path[0]] -= q
        self.last = {key:max(0.,available.get(key,0.)) for key in stock}
        return {'flows':flows}
