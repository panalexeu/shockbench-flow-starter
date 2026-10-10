# -1.0129659972829614
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

class Agent:
    def __init__(self, config):
        self.s=config['static']; self.l=config['layout']; self.T=int(config['T'])
        self.tail=self.s['edges']['tail']; self.head=self.s['edges']['head']
        self.names=self.s['commodities']['id']; self.cp={n:i for i,n in enumerate(self.l['chokepoints'])}
        self.routes=[]; self.lookup={}
        for i,(e,k,l) in enumerate(zip(self.s['action_slots']['edge'],self.s['action_slots']['k'],self.s['action_slots']['lane'])):
            lane=int(l) if l is not None and int(l)>=0 else -1
            es=list(self.s['lanes']['edges'][lane]) if lane>=0 else [int(e)]
            self.routes.append((self.tail[es[0]],self.head[es[-1]],int(k),es,lane))
            self.lookup[(lane,int(k)) if lane>=0 else (-1,int(e),int(k))]=i
        pairs=set(map(tuple,self.l['stock_slots']))|set(map(tuple,self.l['supply_slots']))|set(map(tuple,self.l['demands']))
        for a,b,k,es,l in self.routes: pairs.update([(a,k),(b,k)])
        self.pairs=sorted(pairs); self.idx={p:i for i,p in enumerate(self.pairs)}
        self.rate=np.zeros(len(self.pairs)); self.base=np.zeros(len(self.routes)); self.prev=None; self.started=False
        self.sources=set(map(tuple,self.l['supply_slots']))

    def act(self,o):
        def read(key,default):
            d=np.asarray(default); x=np.asarray(o.get(key,d))
            return np.where(o.get(key+'.observed',np.ones_like(x)),x,d)
        A=len(self.routes); P=len(self.pairs); E=len(self.tail); K=len(self.names); week=int(o['week'][0]); remaining=self.T-week+1
        tau=read('graph_now.tau',self.s['edges']['tau0']).astype(int)
        cap=read('graph_now.u',[0 if x is None else x for x in self.s['edges']['u0']]).astype(float)
        freight=read('graph_now.c',self.s['edges']['c0']); tariff=read('graph_now.tariff',np.zeros((E,K)))
        opened=read('graph_now.open',np.ones(len(self.cp)))
        stock=np.zeros(P); supply=np.zeros(P); inbound=np.zeros(P); arrivals=np.zeros(P); nominal=np.zeros(A)
        for p,q in zip(self.l['stock_slots'],read('stock.qty',np.zeros(len(self.l['stock_slots'])))): stock[self.idx[tuple(p)]]=max(0,float(q))
        for p,q in zip(self.l['supply_slots'],read('graph_now.supply.avail',np.zeros(len(self.l['supply_slots'])))): supply[self.idx[tuple(p)]]+=max(0,float(q))
        pq=np.asarray(o.get('pipeline.qty',[]))
        for j in np.flatnonzero(o.get('pipeline.qty.observed',np.zeros_like(pq))):
            e=int(o['pipeline.edge'][j]); k=int(o['pipeline.k'][j]); q=max(0,float(pq[j])); aw=int(o['pipeline.arrival_week'][j])
            lane=int(o['pipeline.lane'][j]) if np.asarray(o.get('pipeline.lane.observed',np.zeros_like(pq)))[j] else -1
            es=[e]
            if 0<=lane<len(self.s['lanes']['edges']):
                full=list(self.s['lanes']['edges'][lane]); es=full[full.index(e):] if e in full else full
            else: lane=-1
            p=(self.head[es[-1]],k)
            if p in self.idx: inbound[self.idx[p]]+=q
            direct=(self.head[e],k)
            if aw<=week and direct in self.idx: arrivals[self.idx[direct]]+=q
            key=(lane,k) if lane>=0 else (-1,e,k)
            if key in self.lookup: nominal[self.lookup[key]]+=q
        for key,row in zip(self.l.get('lot_keys',[]),o.get('queue_lots.qty',[])):
            n,k,l,e=key; es=[e]
            if l is not None and int(l)>=0:
                full=list(self.s['lanes']['edges'][int(l)]); es=full[full.index(e):] if e in full else full
            p=(self.head[es[-1]],k)
            if p in self.idx: inbound[self.idx[p]]+=max(0,float(np.sum(row)))
        for j in np.flatnonzero(o.get('wip.qty.observed',np.zeros_like(o.get('wip.qty',[])))):
            p=(int(o['wip.node'][j]),int(o['wip.k'][j]))
            if p in self.idx: inbound[self.idx[p]]+=max(0,float(o['wip.qty'][j]))
        delays=np.array([sum(max(1,int(tau[e])) for e in es) for a,b,k,es,l in self.routes],float)
        if not self.started:
            self.base=nominal/np.maximum(1,delays)
            for a,(src,dst,k,es,l) in enumerate(self.routes): self.rate[self.idx[dst,k]]+=self.base[a]
            for p in self.pairs:
                i=self.idx[p]
                if p[0] in self.l['grids'] and self.rate[i]<=0: self.rate[i]=stock[i]/6
            self.started=True
        if self.prev is not None:
            old,arr,out,src=self.prev
            used=np.maximum(0,old+arr+src-out-stock)
            for p in self.pairs:
                i=self.idx[p]
                if p[0] in self.l['grids'] or (p[0] in self.l['fabs'] and self.names[p[1]]=='wafer') or (p[0] in self.l['osats'] and self.names[p[1]].endswith('_raw')):
                    self.rate[i]=.8*self.rate[i]+.2*used[i]
        forecast=read('demand_forecast.qty',np.zeros_like(o['demand_forecast.qty']))
        backlog=read('backlog.qty',np.zeros(len(self.l['demands'])))
        penalty=np.zeros(P); rates=self.rate.copy(); extra=np.zeros(P)
        for d,p in enumerate(self.l['demands']):
            i=self.idx[tuple(p)]; rates[i]=max(0,float(np.mean(forecast[d]))); extra[i]=max(0,float(backlog[d]))
        for n,k,v in zip(self.s['sinks']['node'],self.s['sinks']['k'],self.s['sinks']['pi']): penalty[self.idx[n,k]]=float(v)
        reference=max(1000,float(np.max(penalty,initial=0)))
        for p in self.pairs:
            if p[0] in self.l['grids']: penalty[self.idx[p]]=reference
        for _ in range(P):
            old=penalty.copy()
            for a,b,k,es,l in self.routes: penalty[self.idx[a,k]]=max(penalty[self.idx[a,k]],.95*old[self.idx[b,k]])
            for n in self.l['osats']:
                for p in self.pairs:
                    if p[0]==n and self.names[p[1]].endswith('_raw'):
                        name=self.names[p[1]][:-4]
                        if name in self.names and (n,self.names.index(name)) in self.idx: penalty[self.idx[p]]=max(penalty[self.idx[p]],.85*old[self.idx[n,self.names.index(name)]])
            for n in self.l['fabs']:
                if 'wafer' in self.names and (n,self.names.index('wafer')) in self.idx:
                    outs=[old[self.idx[p]] for p in self.pairs if p[0]==n and self.names[p[1]].endswith('_raw')]
                    penalty[self.idx[n,self.names.index('wafer')]]=max(penalty[self.idx[n,self.names.index('wafer')]],.75*max(outs,default=0))
            if np.allclose(old,penalty): break
        incoming=[[] for _ in range(P)]; outgoing=[[] for _ in range(P)]; resources={}; c=[]; bounds=[]; lead=np.zeros(P)
        mask=o.get('action_mask',np.ones(A))
        for a,(src,dst,k,es,l) in enumerate(self.routes):
            i=self.idx[dst,k]; incoming[i].append(a); outgoing[self.idx[src,k]].append(a)
            if self.base[a]>0: lead[i]+=self.base[a]*delays[a]
            price=sum(float(freight[e])+float(tariff[e,k])*float(self.s['commodities']['v'][k]) for e in es)
            cps={self.tail[e] for e in es if self.tail[e] in self.cp}
            valid=bool(mask[a]) and delays[a]<remaining and all(opened[self.cp[n]]>0 for n in cps)
            c.append(price); bounds.append((0,max(0,min(cap[e] for e in es)) if valid else 0))
            for e in es: resources.setdefault(('edge',e),[]).append(a)
            for n in cps: resources.setdefault((self.s['commodities']['pool'][k],n),[]).append(a)
        rr=[]; cc=[]; vv=[]; rhs=[]
        def ub(items,b):
            r=len(rhs); rhs.append(float(b))
            for j,v in items: rr.append(r); cc.append(j); vv.append(v)
        for i,p in enumerate(self.pairs):
            ub([(a,1) for a in outgoing[i]],stock[i]+supply[i])
            if not incoming[i] or p in self.sources or penalty[i]<=0: continue
            total=sum(self.base[a] for a in incoming[i])
            lt=lead[i]/total if total>0 else min(delays[a] for a in incoming[i])
            rate=rates[i]
            if rate<=0: rate=total
            target=min(remaining*rate,rate*(lt+1.25))+extra[i]
            critical=min(target,rate*(lt+.25)+extra[i])
            pos=stock[i]+inbound[i]+supply[i]
            items=[(a,1) for a in outgoing[i]]+[(a,-1) for a in incoming[i]]
            j=len(c); c.append(max(1,.8*penalty[i])); bounds.append((0,None)); ub(items+[(j,-1)],pos-critical)
            j=len(c); c.append(max(.1,.08*penalty[i])); bounds.append((0,None)); ub(items+[(j,-1)],pos-target)
        for (typ,n),js in resources.items():
            limit=cap[n] if typ=='edge' else read('graph_now.kappa.'+typ,np.full(len(self.cp),1e12))[self.cp[n]]*opened[self.cp[n]]
            ub([(j,1) for j in js],max(0,float(limit)))
        res=linprog(c,A_ub=coo_matrix((vv,(rr,cc)),shape=(len(rhs),len(c))).tocsr(),b_ub=rhs,bounds=bounds,method='highs',options={'time_limit':8})
        flows=np.zeros(A) if res.x is None else np.maximum(0,res.x[:A])
        out=np.zeros(P)
        for a,(src,dst,k,es,l) in enumerate(self.routes): out[self.idx[src,k]]+=flows[a]
        self.prev=(stock.copy(),arrivals.copy(),out,supply.copy())
        return {'flows':flows}
