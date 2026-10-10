# 0.5412517726069141
import numpy as np

class Agent:
    def __init__(self, config):
        self.s, self.l = config['static'], config['layout']
        self.T = int(config['T'])
        e, s, l = self.s['edges'], self.s, self.l
        self.tail = np.asarray(e['tail'], int)
        self.head = np.asarray(e['head'], int)
        self.u0 = np.asarray([0. if x is None else x for x in e['u0']], float)
        self.c0 = np.asarray([0. if x is None else x for x in e['c0']], float)
        self.t0 = np.asarray([1. if x is None else x for x in e['tau0']], float)
        self.values = np.asarray(s['commodities']['v'], float)
        self.pool = s['commodities'].get('pool', ['tb'] * len(self.values))
        self.stocks = [tuple(map(int, x)) for x in l['stock_slots']]
        self.supplies = [tuple(map(int, x)) for x in l['supply_slots']]
        self.demands = [tuple(map(int, x)) for x in l['demands']]
        self.di = {p:i for i,p in enumerate(self.demands)}
        self.cp = {int(n):i for i,n in enumerate(l['chokepoints'])}
        self.routes = []
        grouped, reverse = {}, {}
        a = s['action_slots']
        for i, (edge, k, lane) in enumerate(zip(a['edge'], a['k'], a['lane'])):
            edge, k = int(edge), int(k)
            lane = -1 if lane is None else int(lane)
            es = [edge] if lane < 0 else list(map(int, s['lanes']['edges'][lane]))
            if not es:
                continue
            cps = [] if lane < 0 else list(map(int, s['lanes']['chokepoints'][lane]))
            src, dst = int(self.tail[es[0]]), int(self.head[es[-1]])
            self.routes.append({'i':i, 'k':k, 'es':es, 'cps':cps, 'src':src, 'dst':dst})
            reverse.setdefault((dst,k), set()).add(src)
            alt = e['alt_of'][edge] if lane < 0 else s['lanes']['alt_of'][lane]
            grouped.setdefault((src,k,dst), []).append((alt, min(self.u0[x] for x in es)))
        incoming, outgoing = {}, {}
        for (src,k,dst), choices in grouped.items():
            base = [cap for alt,cap in choices if alt is None]
            cap = sum(base) if base else max((cap for _,cap in choices), default=0.)
            incoming[(dst,k)] = incoming.get((dst,k),0.) + cap
            outgoing[(src,k)] = outgoing.get((src,k),0.) + cap
        self.rate = {}
        for p in set(incoming) | set(outgoing):
            x, y = incoming.get(p), outgoing.get(p)
            self.rate[p] = min(x,y) if x is not None and y is not None else max(x or 0.,y or 0.)
        self.pi = {(int(n),int(k)):float(pi) for n,k,pi in zip(s['sinks']['node'],s['sinks']['k'],s['sinks']['pi'])}
        self.maxpi = max([1.] + list(self.pi.values()))
        self.priority, self.desc = {}, {}
        for di, (sink,k) in enumerate(self.demands):
            q, seen = [(sink,0)], {sink}
            for node, dist in q:
                p = (node,k)
                self.priority[p] = max(self.priority.get(p,0.), self.pi.get(p,1.)/(1.+.12*dist))
                self.desc.setdefault(p, set()).add(di)
                for prev in reverse.get(p, ()):
                    if prev not in seen:
                        seen.add(prev)
                        q.append((prev,dist+1))

    @staticmethod
    def field(o, key, fallback):
        fb = np.asarray(fallback)
        if key not in o:
            return fb.copy()
        x = np.asarray(o[key])
        m = o.get(key+'.observed')
        return np.where(m,x,fb) if m is not None and np.shape(m)==x.shape else x.copy()

    def act(self, o):
        w = int(np.asarray(o['week']).reshape(-1)[0])
        ne, nk = len(self.tail), len(self.values)
        ns = len(self.s['action_slots']['edge'])
        u = self.field(o,'graph_now.u',self.u0).astype(float)
        c = self.field(o,'graph_now.c',self.c0).astype(float)
        tau = np.maximum(1.,self.field(o,'graph_now.tau',self.t0).astype(float))
        tariff = self.field(o,'graph_now.tariff',np.zeros((ne,nk))).astype(float)
        banned = self.field(o,'graph_now.prohibited',np.zeros((ne,nk))).astype(bool)
        opened = self.field(o,'graph_now.open',np.ones(len(self.cp))).astype(float)
        mask = np.asarray(o.get('action_mask',np.ones(ns))).reshape(-1)
        warnings = {}
        ws = self.field(o,'warning.score',np.zeros(len(self.l['warning_units']))).reshape(-1)
        for key, val in zip(self.l['warning_units'], ws):
            warnings[tuple(key)] = float(np.clip(val,0.,1.))

        stock_left, supply_left = {}, {}
        for p,q in zip(self.stocks,self.field(o,'stock.qty',np.zeros(len(self.stocks))).reshape(-1)):
            stock_left[p] = max(0.,float(q))
        for p,q in zip(self.supplies,self.field(o,'graph_now.supply.avail',np.zeros(len(self.supplies))).reshape(-1)):
            supply_left[p] = supply_left.get(p,0.) + max(0.,float(q))
        projected = {}
        def add(p,t,q):
            if q > 0:
                projected.setdefault(p,[]).append((int(t),float(q)))
        lanes = self.s['lanes']['edges']
        def lane_info(edge,lane):
            es = list(map(int,lanes[lane])) if lane is not None and 0 <= int(lane) < len(lanes) else [int(edge)]
            if edge not in es:
                es = [int(edge)]
            pos = es.index(int(edge))
            return int(self.head[es[-1]]), es[pos:]

        pq = np.asarray(o.get('pipeline.qty',[]),float).reshape(-1)
        pm = np.asarray(o.get('pipeline.qty.observed',np.ones(pq.size)),bool).reshape(-1)
        pe = np.asarray(o.get('pipeline.edge',np.zeros(pq.size)),int).reshape(-1)
        pk = np.asarray(o.get('pipeline.k',np.zeros(pq.size)),int).reshape(-1)
        pl = np.asarray(o.get('pipeline.lane',np.full(pq.size,-1)),int).reshape(-1)
        plm = np.asarray(o.get('pipeline.lane.observed',np.zeros(pq.size)),bool).reshape(-1)
        pa = np.asarray(o.get('pipeline.arrival_week',np.full(pq.size,w)),int).reshape(-1)
        for j,q in enumerate(pq):
            if j>=pm.size or not pm[j] or q<=0 or j>=pe.size or not 0<=pe[j]<ne:
                continue
            lane = int(pl[j]) if j<pl.size and j<plm.size and plm[j] else None
            dst, rem = lane_info(pe[j],lane)
            add((dst,int(pk[j])),max(w,int(pa[j]))+int(np.ceil(sum(tau[x] for x in rem[1:]))),q)
        wq = np.asarray(o.get('wip.qty',[]),float).reshape(-1)
        wm = np.asarray(o.get('wip.qty.observed',np.ones(wq.size)),bool).reshape(-1)
        for j,q in enumerate(wq):
            if j<wm.size and wm[j] and q>0:
                add((int(o['wip.node'][j]),int(o['wip.k'][j])),max(w,int(o['wip.out_week'][j])),q)

        ends = {}
        ce = np.asarray(o.get('closure_end.chokepoint',[]),int).reshape(-1)
        cem = np.asarray(o.get('closure_end.chokepoint.observed',np.zeros(ce.size)),bool).reshape(-1)
        ew = np.asarray(o.get('closure_end.end_week',np.zeros(ce.size)),int).reshape(-1)
        ewm = np.asarray(o.get('closure_end.end_week.observed',np.zeros(ce.size)),bool).reshape(-1)
        for j in range(min(ce.size,cem.size,ew.size,ewm.size)):
            if cem[j] and ewm[j]:
                ends[int(ce[j])] = int(ew[j])+1
        lots = np.asarray(o.get('queue_lots.qty',np.zeros((0,0))),float)
        lm = np.asarray(o.get('queue_lots.qty.observed',np.ones(lots.shape)),bool)
        for row,key in enumerate(self.l.get('lot_keys',[])[:lots.shape[0]]):
            if len(key)<4 or row>=lm.shape[0]:
                continue
            cp,k,lane,edge = key
            q = float(np.sum(np.where(lm[row],lots[row],0.)))
            if q<=0 or not 0<=int(edge)<ne:
                continue
            dst, rem = lane_info(int(edge),lane)
            start = w
            ci = self.cp.get(int(cp))
            if ci is not None and opened[ci]<=.001:
                start = max(start,ends.get(int(cp),w+3))
            add((dst,int(k)),start+int(np.ceil(sum(tau[x] for x in rem))),q)

        pending = {}
        p_e = np.asarray(o.get('pending_prohibitions.edge',[]),int).reshape(-1)
        p_m = np.asarray(o.get('pending_prohibitions.edge.observed',np.zeros(p_e.size)),bool).reshape(-1)
        p_k = np.asarray(o.get('pending_prohibitions.k',np.zeros(p_e.size)),int).reshape(-1)
        p_w = np.asarray(o.get('pending_prohibitions.effective_week',np.full(p_e.size,self.T+1)),int).reshape(-1)
        for j in range(min(p_e.size,p_m.size,p_k.size,p_w.size)):
            if p_m[j]:
                key = (int(p_e[j]),int(p_k[j]))
                pending[key] = min(pending.get(key,self.T+1),int(p_w[j]))

        forecast = np.asarray(o.get('demand_forecast.qty',np.zeros((len(self.demands),1))),float)
        backlog = np.asarray(o.get('backlog.qty',np.zeros(len(self.demands))),float).reshape(-1)
        sink_rate = []
        for di in range(len(self.demands)):
            row = forecast[di] if forecast.ndim>1 and di<forecast.shape[0] else np.zeros(0)
            sink_rate.append(max(0.,float(np.mean(row[:min(8,len(row))])) if len(row) else 0.))
        def coverage(p,cut):
            return stock_left.get(p,0.) + sum(q for t,q in projected.get(p,()) if t<=cut)
        def requirement(r,lead,risk):
            p = (r['dst'],r['k'])
            remain = max(1,self.T-w+1)
            if p in self.di:
                di = self.di[p]
                h = min(remain,max(1,int(np.ceil(lead))+2+int(np.ceil(risk))))
                row = forecast[di] if forecast.ndim>1 and di<forecast.shape[0] else np.zeros(0)
                n = min(h,len(row))
                total = float(backlog[di]) if di<backlog.size else 0.
                if n:
                    total += float(np.sum(row[:n]))
                    if h>n:
                        total += (h-n)*float(np.mean(row[-min(3,n):]))
                return total,w+h-1
            dur = min(float(remain),max(1.,lead+2.+2.*risk))
            rate = max(0.,float(self.rate.get(p,0.)))
            desc = self.desc.get(p,())
            if desc:
                downstream = sum(sink_rate[j] for j in desc if j<len(sink_rate))
                if downstream>0:
                    rate = min(rate,downstream) if rate>0 else downstream
            return rate*dur,w+int(np.ceil(dur))

        edge_left = np.maximum(0.,u.copy())
        cp_left = {}
        for pool in ('tb','ct'):
            caps = self.field(o,'graph_now.kappa.'+pool,np.full(len(self.cp),np.inf)).astype(float)
            for cp,ci in self.cp.items():
                cap = caps[ci] if ci<len(caps) else np.inf
                cp_left[(cp,pool)] = max(0.,float(cap)*max(0.,opened[ci]))
        def exposure(r):
            vals = [warnings.get(('chokepoint',cp),0.) for cp in r['cps']]
            regions = []
            nr = self.s['nodes'].get('region',[])
            for node in (r['src'],r['dst']):
                if 0<=node<len(nr):
                    regions.append(int(nr[node]))
            vals.extend(warnings.get(('region',x),0.) for x in regions)
            dy = self.s.get('dyads',{})
            for j,(a,b) in enumerate(zip(dy.get('a',[]),dy.get('b',[]))):
                if int(a) in regions and int(b) in regions:
                    vals.append(warnings.get(('dyad',j),0.))
            return max(vals,default=0.)

        candidates = []
        for r in self.routes:
            i,k,es,cps = r['i'],r['k'],r['es'],r['cps']
            if i>=mask.size or mask[i]<=0 or any(banned[e,k] for e in es):
                continue
            lead = float(sum(tau[e] for e in es))
            if w+lead>self.T+1:
                continue
            elapsed, invalid = 0., False
            for e in es:
                if pending.get((e,k),self.T+1)<=w+elapsed:
                    invalid = True
                    break
                elapsed += tau[e]
            if invalid:
                continue
            risk = exposure(r)
            cap = min((max(0.,float(u[e])) for e in es),default=0.)
            for cp in cps:
                cap = min(cap,cp_left.get((cp,self.pool[k]),0.))
            if cap<=0:
                continue
            target,cut = requirement(r,lead,risk)
            need = max(0.,target-coverage((r['dst'],k),cut))
            if need<=0:
                continue
            freight = sum(float(c[e])+float(tariff[e,k])*float(self.values[k]) for e in es)
            freight = freight*(1.+.35*risk)+.002*float(self.values[k])*lead
            priority = max(1.,float(self.priority.get((r['dst'],k),self.maxpi*.25)))
            urgency = .35+.65*min(1.,need/max(target,1e-9))
            candidates.append((freight/(priority*urgency),lead,i,cap,r,risk))

        flows = np.zeros(ns,float)
        for _,lead,i,planned,r,risk in sorted(candidates,key=lambda x:(x[0],x[1],x[2])):
            k,es,cps = r['k'],r['es'],r['cps']
            srcp,dstp = (r['src'],k),(r['dst'],k)
            target,cut = requirement(r,lead,risk)
            need = max(0.,target-coverage(dstp,cut))
            available = stock_left.get(srcp,0.)+supply_left.get(srcp,0.)
            q = min(planned,need,available)
            for e in es:
                q = min(q,max(0.,edge_left[e]))
            for cp in cps:
                q = min(q,cp_left.get((cp,self.pool[k]),0.))
            if q<=0 or not np.isfinite(q):
                continue
            flows[i] = q
            used = min(q,stock_left.get(srcp,0.))
            stock_left[srcp] = max(0.,stock_left.get(srcp,0.)-used)
            supply_left[srcp] = max(0.,supply_left.get(srcp,0.)-(q-used))
            for e in es:
                edge_left[e] = max(0.,edge_left[e]-q)
            for cp in cps:
                key = (cp,self.pool[k])
                if key in cp_left:
                    cp_left[key] = max(0.,cp_left[key]-q)
            add(dstp,w+int(np.ceil(lead)),q)
        return {'flows':flows}
