# 0.15664234109904618
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

class Agent:
    def __init__(self, config):
        self.s=config['static']; self.l=config['layout']; self.T=int(config['T'])
        self.names=self.s['commodities']['id']; self.tail=self.s['edges']['tail']; self.head=self.s['edges']['head']
        self.cp={n:i for i,n in enumerate(self.l['chokepoints'])}
        self.routes=[]
        for e,k,lane in zip(self.s['action_slots']['edge'],self.s['action_slots']['k'],self.s['action_slots']['lane']):
            es=list(self.s['lanes']['edges'][int(lane)]) if lane is not None and int(lane)>=0 else [int(e)]
            self.routes.append((self.tail[es[0]],self.head[es[-1]],int(k),es))
        pairs=set(map(tuple,self.l['stock_slots']))|set(map(tuple,self.l['supply_slots']))|set(map(tuple,self.l['demands']))
        for a,b,k,es in self.routes: pairs.update([(a,k),(b,k)])
        self.pairs=sorted(pairs); self.idx={p:i for i,p in enumerate(self.pairs)}
        self.sources=set(map(tuple,self.l['supply_slots']))
        self.incoming=[[] for p in self.pairs]; self.outgoing=[[] for p in self.pairs]
        for j,(a,b,k,es) in enumerate(self.routes):
            self.incoming[self.idx[b,k]].append(j); self.outgoing[self.idx[a,k]].append(j)
        self.started=False; self.prev=None
        self.conversions=[]
        for n in self.l['osats']:
            for nn,k in self.pairs:
                if nn==n and self.names[k].endswith('_raw') and self.names[k][:-4] in self.names:
                    out=self.names.index(self.names[k][:-4])
                    if (n,out) in self.idx: self.conversions.append(((n,k),(n,out),1))
        for n in self.l['fabs']:
            if 'wafer' in self.names and (n,self.names.index('wafer')) in self.idx:
                for nn,k in self.pairs:
                    if nn==n and self.names[k].endswith('_raw'): self.conversions.append(((n,self.names.index('wafer')),(n,k),2))

    def act(self,o):
        def read(key,default):
            d=np.asarray(default); x=np.asarray(o.get(key,d))
            return np.where(o.get(key+'.observed',np.ones_like(x)),x,d)
        week=int(o['week'][0]); remaining=self.T-week+1
        P=len(self.pairs); A=len(self.routes); E=len(self.tail); K=len(self.names)
        cap=read('graph_now.u',[0 if x is None else x for x in self.s['edges']['u0']]).astype(float)
        tau=read('graph_now.tau',self.s['edges']['tau0']).astype(int)
        cost=read('graph_now.c',self.s['edges']['c0']); tariff=read('graph_now.tariff',np.zeros((E,K)))
        opened=read('graph_now.open',np.ones(len(self.cp)))
        stock=np.zeros(P); supply=np.zeros(P); pipe=np.zeros(P); arrivals=np.zeros(P)
        for p,q in zip(self.l['stock_slots'],read('stock.qty',np.zeros(len(self.l['stock_slots'])))): stock[self.idx[tuple(p)]]=max(0,float(q))
        for p,q in zip(self.l['supply_slots'],read('graph_now.supply.avail',np.zeros(len(self.l['supply_slots'])))): supply[self.idx[tuple(p)]]+=max(0,float(q))
        for j in np.flatnonzero(o.get('pipeline.qty.observed',np.zeros_like(o.get('pipeline.qty',[])))):
            e=int(o['pipeline.edge'][j]); k=int(o['pipeline.k'][j]); q=max(0,float(o['pipeline.qty'][j])); aw=int(o['pipeline.arrival_week'][j]); es=[e]
            lm=o.get('pipeline.lane.observed',np.zeros_like(o['pipeline.qty']))
            lane=int(o['pipeline.lane'][j]) if lm[j] else -1
            if 0<=lane<len(self.s['lanes']['edges']):
                full=list(self.s['lanes']['edges'][lane]); es=full[full.index(e):] if e in full else full
            p=(self.head[es[-1]],k)
            if p in self.idx: pipe[self.idx[p]]+=q
            p=(self.head[e],k)
            if aw<=week and p in self.idx: arrivals[self.idx[p]]+=q
        for key,row in zip(self.l.get('lot_keys',[]),o.get('queue_lots.qty',[])):
            n,k,lane,e=key; es=[e]
            if lane is not None and int(lane)>=0:
                full=list(self.s['lanes']['edges'][int(lane)]); es=full[full.index(e):] if e in full else full
            p=(self.head[es[-1]],k)
            if p in self.idx: pipe[self.idx[p]]+=max(0,float(np.sum(row)))
        for j in np.flatnonzero(o.get('wip.qty.observed',np.zeros_like(o.get('wip.qty',[])))):
            p=(int(o['wip.node'][j]),int(o['wip.k'][j])); q=max(0,float(o['wip.qty'][j]))
            if p in self.idx:
                pipe[self.idx[p]]+=q
                if int(o['wip.out_week'][j])<=week: arrivals[self.idx[p]]+=q
        delays=np.array([sum(max(1,int(tau[e])) for e in es) for a,b,k,es in self.routes])
        forecast=read('demand_forecast.qty',np.zeros_like(o['demand_forecast.qty']))
        if not self.started:
            self.rate=np.zeros(P); self.lead=np.ones(P)
            for i in range(P):
                js=self.incoming[i]
                if js:
                    ordinary=[j for j in js if not any(self.s['edges']['alt_of'][e] is not None for e in self.routes[j][3])]
                    self.lead[i]=min(delays[j] for j in (ordinary or js))
                    self.rate[i]=pipe[i]/max(1,self.lead[i])
                if self.rate[i]==0 and self.pairs[i][0] in self.l['grids']: self.rate[i]=stock[i]/5
            self.initial_rate=self.rate.copy(); self.initial_stock=stock.copy()
            self.target=stock+pipe+self.rate
            self.started=True
        if self.prev is not None:
            old,arr,src=self.prev
            executed=read('last_week.clip.executed',np.zeros(A)); out=np.zeros(P)
            for j,(a,b,k,es) in enumerate(self.routes): out[self.idx[a,k]]+=max(0,executed[j])
            used=np.maximum(0,old+arr+src-out-stock)
            for i,p in enumerate(self.pairs):
                if p[0] in self.l['grids'] or any(p==x for x,y,d in self.conversions):
                    self.rate[i]=.9*self.rate[i]+.1*used[i]
        value=np.zeros(P); downstream=np.full(P,np.inf)
        for n,k,v in zip(self.s['sinks']['node'],self.s['sinks']['k'],self.s['sinks']['pi']): value[self.idx[n,k]]=float(v); downstream[self.idx[n,k]]=0
        for i,p in enumerate(self.pairs):
            if p[0] in self.l['grids']: value[i]=5000; downstream[i]=0
        for _ in range(P):
            old=value.copy(); od=downstream.copy()
            for j,(a,b,k,es) in enumerate(self.routes):
                ia=self.idx[a,k]; ib=self.idx[b,k]
                value[ia]=max(value[ia],.98*old[ib]); downstream[ia]=min(downstream[ia],delays[j]+od[ib])
            for x,y,d in self.conversions:
                value[self.idx[x]]=max(value[self.idx[x]],.85*old[self.idx[y]])
                downstream[self.idx[x]]=min(downstream[self.idx[x]],d+od[self.idx[y]])
            if np.array_equal(old,value) and np.array_equal(od,downstream): break
        target=self.target.copy()
        for i,p in enumerate(self.pairs):
            if self.initial_rate[i]>0:
                target[i]=self.initial_stock[i]+(self.lead[i]+1)*max(self.rate[i],.7*self.initial_rate[i])
            target[i]=min(target[i],max(0,remaining-downstream[i])*max(self.rate[i],self.initial_rate[i])) if np.isfinite(downstream[i]) else target[i]
        backlog=read('backlog.qty',np.zeros(len(self.l['demands'])))
        for d,p in enumerate(self.l['demands']):
            i=self.idx[tuple(p)]; rate=max(0,float(np.mean(forecast[d])))
            target[i]=min(remaining*rate,(self.lead[i]+1.5)*rate)+max(0,float(backlog[d]))
        c=[]; bounds=[]; resources={}; mask=o.get('action_mask',np.ones(A))
        for j,(a,b,k,es) in enumerate(self.routes):
            cps={self.tail[e] for e in es if self.tail[e] in self.cp}
            valid=bool(mask[j]) and delays[j]+downstream[self.idx[b,k]]<remaining and all(opened[self.cp[n]]>0 for n in cps)
            c.append(sum(float(cost[e])+float(tariff[e,k])*float(self.s['commodities']['v'][k]) for e in es))
            bounds.append((0,max(0,min(cap[e] for e in es)) if valid else 0))
            for e in es: resources.setdefault(('edge',e),[]).append(j)
            for n in cps: resources.setdefault((self.s['commodities']['pool'][k],n),[]).append(j)
        rr=[]; cc=[]; vv=[]; rhs=[]
        def ub(items,b):
            r=len(rhs); rhs.append(float(b))
            for j,v in items: rr.append(r); cc.append(j); vv.append(v)
        for i,p in enumerate(self.pairs):
            ub([(j,1) for j in self.outgoing[i]],stock[i]+supply[i])
            if p in self.sources or not self.incoming[i]: continue
            j=len(c); c.append(max(1,value[i])); bounds.append((0,None))
            ub([(a,1) for a in self.outgoing[i]]+[(a,-1) for a in self.incoming[i]]+[(j,-1)],stock[i]+pipe[i]+supply[i]-target[i])
        for (typ,n),js in resources.items():
            lim=cap[n] if typ=='edge' else read('graph_now.kappa.'+typ,np.full(len(self.cp),1e12))[self.cp[n]]*opened[self.cp[n]]
            ub([(j,1) for j in js],max(0,float(lim)))
        result=linprog(c,A_ub=coo_matrix((vv,(rr,cc)),shape=(len(rhs),len(c))).tocsr(),b_ub=rhs,bounds=bounds,method='highs',options={'time_limit':8})
        flows=np.zeros(A) if result.x is None else np.maximum(0,result.x[:A])
        self.prev=(stock.copy(),arrivals.copy(),supply.copy())
        return {'flows':flows}
