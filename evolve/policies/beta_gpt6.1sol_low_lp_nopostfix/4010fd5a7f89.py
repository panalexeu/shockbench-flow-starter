# -0.9428338994169324
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

class Agent:
    def __init__(self, config):
        self.s = config['static']; self.l = config['layout']; self.T = int(config['T'])
        self.names = self.s['commodities']['id']
        self.tail = self.s['edges']['tail']; self.head = self.s['edges']['head']
        self.cp = {n:i for i,n in enumerate(self.l['chokepoints'])}
        self.routes = []
        for e,k,lane in zip(self.s['action_slots']['edge'], self.s['action_slots']['k'], self.s['action_slots']['lane']):
            es = list(self.s['lanes']['edges'][int(lane)]) if lane is not None and int(lane)>=0 else [e]
            self.routes.append((self.tail[es[0]],self.head[es[-1]],k,es))
        pairs = set(map(tuple,self.l['stock_slots'])) | set(map(tuple,self.l['supply_slots'])) | set(map(tuple,self.l['demands']))
        for a,b,k,es in self.routes: pairs.update([(a,k),(b,k)])
        self.pairs = sorted(pairs); self.idx = {p:i for i,p in enumerate(self.pairs)}
        self.sources = set(map(tuple,self.l['supply_slots']))
        self.prod = []
        for n in self.l['fabs']:
            for p in self.pairs:
                if p[0]==n and str(self.names[p[1]]).endswith('_raw'):
                    if 'wafer' in self.names and (n,self.names.index('wafer')) in self.idx:
                        self.prod.append((n,self.names.index('wafer'),p[1],'fab'))
        for n in self.l['osats']:
            for nn,k in self.pairs:
                if nn==n and str(self.names[k]).endswith('_raw'):
                    name=self.names[k][:-4]
                    if name in self.names and (n,self.names.index(name)) in self.idx:
                        self.prod.append((n,k,self.names.index(name),'osat'))
        self.fuel = {}; self.initialized = False

    def act(self,o):
        def read(key,default):
            default=np.asarray(default)
            x=np.asarray(o.get(key,default))
            return np.where(np.asarray(o.get(key+'.observed',np.ones_like(x))),x,default)
        A=len(self.routes); P=len(self.pairs); week=int(o['week'][0])
        E=len(self.tail); K=len(self.names)
        cap=read('graph_now.u',[0 if x is None else x for x in self.s['edges']['u0']]).astype(float)
        tau=read('graph_now.tau',self.s['edges']['tau0'])
        freight=read('graph_now.c',self.s['edges']['c0'])
        tariff=read('graph_now.tariff',np.zeros((E,K)))
        opened=read('graph_now.open',np.ones(len(self.cp)))
        stock=np.zeros(P)
        for p,q in zip(self.l['stock_slots'],read('stock.qty',np.zeros(len(self.l['stock_slots'])))):
            stock[self.idx[tuple(p)]]=max(0,q)
        supply=np.zeros(P)
        for p,q in zip(self.l['supply_slots'],read('graph_now.supply.avail',np.zeros(len(self.l['supply_slots'])))):
            supply[self.idx[tuple(p)]]+=max(0,q)
        inbound=np.zeros(P); inbound_time=np.zeros(P)
        pipe_total={}; pipe_delay={}
        for j in np.flatnonzero(o.get('pipeline.qty.observed',np.zeros_like(o.get('pipeline.qty',[])))):
            e=int(o['pipeline.edge'][j]); k=int(o['pipeline.k'][j]); q=float(o['pipeline.qty'][j]); aw=int(o['pipeline.arrival_week'][j])
            es=[e]; lane=-1
            if o.get('pipeline.lane.observed',np.zeros_like(o['pipeline.qty']))[j]: lane=int(o['pipeline.lane'][j])
            full=es
            if 0<=lane<len(self.s['lanes']['edges']):
                full=list(self.s['lanes']['edges'][lane])
                if e in full:
                    es=full[full.index(e):]
                    aw+=sum(max(1,int(tau[z])) for z in es[1:])
            p=(self.head[es[-1]],k)
            if p in self.idx:
                i=self.idx[p]; inbound[i]+=q; inbound_time[i]+=q*max(0,aw-week)
                pipe_total[p]=pipe_total.get(p,0)+q
                pipe_delay[p]=max(pipe_delay.get(p,1),sum(max(1,int(tau[z])) for z in full))
        if not self.initialized:
            for p in self.pairs:
                if p[0] in self.l['grids'] and self.names[p[1]] in ('crude','lng','nucfuel'):
                    rate=pipe_total.get(p,0)/pipe_delay.get(p,1)
                    if rate<=0: rate=stock[self.idx[p]]/6
                    self.fuel[p]=max(0,rate)
            self.initialized=True
        for j in np.flatnonzero(o.get('wip.qty.observed',np.zeros_like(o.get('wip.qty',[])))):
            p=(int(o['wip.node'][j]),int(o['wip.k'][j]))
            if p in self.idx: inbound[self.idx[p]]+=max(0,float(o['wip.qty'][j]))
        for key,row in zip(self.l.get('lot_keys',[]),o.get('queue_lots.qty',[])):
            n,k,lane,e=key; es=[e]
            if lane is not None and int(lane)>=0:
                route=list(self.s['lanes']['edges'][int(lane)])
                if e in route: es=route[route.index(e):]
            p=(self.head[es[-1]],k)
            if p in self.idx: inbound[self.idx[p]]+=max(0,float(np.sum(row)))
        forecast=read('demand_forecast.qty',np.zeros_like(o['demand_forecast.qty']))
        pi={(n,k):float(v) for n,k,v in zip(self.s['sinks']['node'],self.s['sinks']['k'],self.s['sinks']['pi'])}
        demand=np.zeros(P); penalty=np.zeros(P)
        backlog=read('backlog.qty',np.zeros(len(self.l['demands'])))
        for d,p in enumerate(self.l['demands']):
            p=tuple(p); i=self.idx[p]
            demand[i]=max(0,float(np.mean(forecast[d,:min(4,forecast.shape[1])]))+float(backlog[d])/4)
            penalty[i]=pi.get(p,10000)
        for p,r in self.fuel.items():
            i=self.idx[p]; demand[i]+=r; penalty[i]=max(penalty[i],10000)
        mask=np.asarray(o.get('action_mask',np.ones(A)))
        prices=np.zeros(A); upper=np.zeros(A); delays=np.zeros(A)
        resources={}
        for a,(src,dst,k,es) in enumerate(self.routes):
            delays[a]=sum(max(1,int(tau[e])) for e in es)
            prices[a]=sum(float(freight[e])+float(tariff[e,k])*self.s['commodities']['v'][k] for e in es)
            cps=set(self.tail[e] for e in es if self.tail[e] in self.cp)
            valid=bool(mask[a]) and all(opened[self.cp[n]]>0 for n in cps)
            upper[a]=max(0,min(cap[e] for e in es)) if valid else 0
            for e in es: resources.setdefault(('edge',e),[]).append(a)
            for n in cps: resources.setdefault((self.s['commodities']['pool'][k],n),[]).append(a)
        limits={}
        for key in resources:
            typ,n=key
            if typ=='edge': limits[key]=max(0,cap[n])
            else: limits[key]=max(0,read('graph_now.kappa.'+typ,np.full(len(self.cp),1e12))[self.cp[n]]*opened[self.cp[n]])
        # Steady-state LP: physical routing and conversion, with soft demand.
        c=list(prices); bounds=[(0,float(v)) for v in upper]
        rows=[]; cols=[]; vals=[]; rhs=[]
        balance=[{} for _ in range(P)]
        def term(i,j,v): balance[i][j]=balance[i].get(j,0)+v
        for a,(src,dst,k,es) in enumerate(self.routes):
            term(self.idx[src,k],a,1); term(self.idx[dst,k],a,-1)
        prodgroups={}
        for n,ik,ok,typ in self.prod:
            j=len(c); c.append(.01); bounds.append((0,None))
            term(self.idx[n,ik],j,1); term(self.idx[n,ok],j,-1)
            prodgroups.setdefault((n,typ),[]).append(j)
        consumption={}
        for i in range(P):
            if demand[i]>0:
                j=len(c); c.append(-penalty[i]); bounds.append((0,float(demand[i]))); term(i,j,1); consumption[i]=j
        def constraint(items,b):
            r=len(rhs); rhs.append(float(b))
            for j,v in items: rows.append(r); cols.append(j); vals.append(v)
        for i in range(P): constraint(balance[i].items(),supply[i])
        for key,js in resources.items(): constraint([(j,1) for j in js],limits[key])
        for (n,typ),js in prodgroups.items():
            ns=self.l['fabs' if typ=='fab' else 'osats']
            key='graph_now.fab.cap_eff' if typ=='fab' else 'graph_now.osat.thr_eff'
            lim=read(key,np.zeros(len(ns)))[ns.index(n)]
            constraint([(j,1) for j in js],max(0,lim))
        N=len(c)
        res=linprog(c,A_ub=coo_matrix((vals,(rows,cols)),shape=(len(rhs),N)).tocsr(),b_ub=rhs,bounds=bounds,method='highs',options={'time_limit':5})
        target=np.zeros(P); value=penalty.copy()
        if res.x is not None:
            for a,(src,dst,k,es) in enumerate(self.routes):
                rate=max(0,res.x[a]); target[self.idx[dst,k]]+=rate*(delays[a]+1.5)
            for n,ik,ok,typ in self.prod:
                i=self.idx[n,ik]
                js=prodgroups.get((n,typ),[])
                target[i]=max(target[i],1.5*sum(res.x[j] for j in js)/max(1,len(js)))
        for i in consumption: target[i]=max(target[i],2*demand[i])
        for _ in range(P):
            old=value.copy()
            for src,dst,k,es in self.routes:
                value[self.idx[src,k]]=max(value[self.idx[src,k]],.96*old[self.idx[dst,k]])
            for n,ik,ok,typ in self.prod:
                value[self.idx[n,ik]]=max(value[self.idx[n,ik]],.9*old[self.idx[n,ok]])
            if np.max(np.abs(value-old),initial=0)<1e-8: break
        # Replenishment LP: never spend stock that is only expected to arrive later.
        c=list(prices); bounds=[(0,float(v)) for v in upper]
        rr=[]; cc=[]; vv=[]; bb=[]
        def ub(items,b):
            r=len(bb); bb.append(float(b))
            for j,v in items: rr.append(r); cc.append(j); vv.append(v)
        outgoing=[[] for _ in range(P)]; incoming=[[] for _ in range(P)]
        for a,(src,dst,k,es) in enumerate(self.routes):
            outgoing[self.idx[src,k]].append(a); incoming[self.idx[dst,k]].append(a)
        for i,p in enumerate(self.pairs):
            ub([(a,1) for a in outgoing[i]],stock[i]+supply[i])
            if p in self.sources or target[i]<=0: continue
            # At the episode end, do not accumulate inventory with no remaining use.
            target[i]=min(target[i],max(0,self.T-week+1)*max(demand[i],target[i]/6))
            j=len(c); c.append(max(1,.65*value[i])); bounds.append((0,None))
            ub([(a,1) for a in outgoing[i]]+[(a,-1) for a in incoming[i]]+[(j,-1)],stock[i]+inbound[i]-target[i])
        for key,js in resources.items(): ub([(a,1) for a in js],limits[key])
        for a in range(A):
            if delays[a]>self.T-week: bounds[a]=(0,0)
        N=len(c)
        res=linprog(c,A_ub=coo_matrix((vv,(rr,cc)),shape=(len(bb),N)).tocsr(),b_ub=bb,bounds=bounds,method='highs',options={'time_limit':5})
        flows=np.zeros(A) if res.x is None else np.maximum(0,res.x[:A])
        return {'flows':flows}
