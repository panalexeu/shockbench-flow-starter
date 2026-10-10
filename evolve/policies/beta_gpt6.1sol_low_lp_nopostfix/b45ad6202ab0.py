# -1.1750083943150118
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

class Agent:
    def __init__(self, config):
        self.s=config['static']; self.l=config['layout']; self.T=int(config['T'])
        self.names=self.s['commodities']['id']; self.tail=self.s['edges']['tail']; self.head=self.s['edges']['head']
        self.cp={n:i for i,n in enumerate(self.l['chokepoints'])}
        self.routes=[]
        for e,k,l in zip(self.s['action_slots']['edge'],self.s['action_slots']['k'],self.s['action_slots']['lane']):
            es=list(self.s['lanes']['edges'][int(l)]) if l is not None and int(l)>=0 else [e]
            self.routes.append((self.tail[es[0]],self.head[es[-1]],k,es))
        pairs=set(map(tuple,self.l['stock_slots']))|set(map(tuple,self.l['supply_slots']))|set(map(tuple,self.l['demands']))
        for a,b,k,es in self.routes: pairs.update([(a,k),(b,k)])
        self.pairs=sorted(pairs); self.idx={p:i for i,p in enumerate(self.pairs)}
        self.prod=[]
        for n in self.l['fabs']:
            if 'wafer' not in self.names: continue
            ik=self.names.index('wafer')
            for nn,k in self.pairs:
                if nn==n and str(self.names[k]).endswith('_raw') and (n,ik) in self.idx: self.prod.append((n,ik,k,'fab',2))
        for n in self.l['osats']:
            for nn,k in self.pairs:
                name=str(self.names[k])
                if nn==n and name.endswith('_raw') and name[:-4] in self.names:
                    ok=self.names.index(name[:-4])
                    if (n,ok) in self.idx: self.prod.append((n,k,ok,'osat',1))
        self.fuel={}; self.prev=None

    def act(self,o):
        def read(key,default):
            d=np.asarray(default); x=np.asarray(o.get(key,d))
            return np.where(o.get(key+'.observed',np.ones_like(x)),x,d)
        week=int(o['week'][0]); H=min(22,self.T-week+1); P=len(self.pairs); A=len(self.routes)
        cap=read('graph_now.u',[0 if v is None else v for v in self.s['edges']['u0']]).astype(float)
        tau=read('graph_now.tau',self.s['edges']['tau0']).astype(int)
        cost=read('graph_now.c',self.s['edges']['c0'])
        tariff=read('graph_now.tariff',np.zeros((len(cap),len(self.names))))
        opened=read('graph_now.open',np.ones(len(self.cp)))
        stock=np.zeros(P)
        for p,q in zip(self.l['stock_slots'],read('stock.qty',np.zeros(len(self.l['stock_slots'])))): stock[self.idx[tuple(p)]]=max(0,q)
        inc=np.zeros((H,P)); direct=np.zeros(P); pipetotal=np.zeros(P)
        for j in np.flatnonzero(o.get('pipeline.qty.observed',np.zeros_like(o.get('pipeline.qty',[])))):
            e=int(o['pipeline.edge'][j]); k=int(o['pipeline.k'][j]); aw=int(o['pipeline.arrival_week'][j]); q=float(o['pipeline.qty'][j]); n=self.head[e]
            lane=int(o['pipeline.lane'][j]) if o.get('pipeline.lane.observed',np.zeros_like(o['pipeline.qty']))[j] else -1
            p=(n,k)
            if p in self.idx and aw<=week: direct[self.idx[p]]+=q
            if 0<=lane<len(self.s['lanes']['edges']):
                es=self.s['lanes']['edges'][lane]
                if e in es:
                    aw+=sum(max(1,int(tau[z])) for z in es[es.index(e)+1:]); n=self.head[es[-1]]
            p=(n,k)
            if p in self.idx:
                i=self.idx[p]; pipetotal[i]+=q; h=max(0,aw-week)
                if h<H: inc[h,i]+=q
        if self.prev is not None:
            old,arr,out=self.prev
            for p in self.fuel:
                i=self.idx[p]; used=max(0,old[i]+arr[i]-out[i]-stock[i])
                if used>0: self.fuel[p]=.7*self.fuel[p]+.3*used
        for p in self.pairs:
            if p[0] in self.l['grids'] and self.names[p[1]] in ('lng','crude','nucfuel') and p not in self.fuel:
                i=self.idx[p]
                ds=[sum(max(1,int(tau[e])) for e in es) for a,b,k,es in self.routes if (b,k)==p]
                self.fuel[p]=max(pipetotal[i]/max(1,min(ds) if ds else 4),stock[i]/8)
        for j in np.flatnonzero(o.get('wip.qty.observed',np.zeros_like(o.get('wip.qty',[])))):
            p=(int(o['wip.node'][j]),int(o['wip.k'][j])); h=max(0,int(o['wip.out_week'][j])-week)
            if p in self.idx and h<H: inc[h,self.idx[p]]+=float(o['wip.qty'][j])
        ends={}
        for j in np.flatnonzero(o.get('closure_end.end_week.observed',[])):
            ends[int(o['closure_end.chokepoint'][j])]=int(o['closure_end.end_week'][j])
        for key,row in zip(self.l.get('lot_keys',[]),o.get('queue_lots.qty',[])):
            n,k,l,e=key; es=[e]
            if l is not None and int(l)>=0:
                full=self.s['lanes']['edges'][int(l)]
                if e in full: es=full[full.index(e):]
            delay=sum(max(1,int(tau[z])) for z in es)
            if opened[self.cp[n]]<=0: delay+=max(1,ends.get(n,week+5)-week)
            p=(self.head[es[-1]],k)
            if p in self.idx and delay<H: inc[delay,self.idx[p]]+=np.sum(row)
        supply=np.zeros(P)
        for p,q in zip(self.l['supply_slots'],read('graph_now.supply.avail',np.zeros(len(self.l['supply_slots'])))): supply[self.idx[tuple(p)]]+=max(0,q)
        inc+=supply
        forecast=read('demand_forecast.qty',np.zeros_like(o['demand_forecast.qty']))
        pi={(n,k):float(v) for n,k,v in zip(self.s['sinks']['node'],self.s['sinks']['k'],self.s['sinks']['pi'])}
        back={(n,k):v for n,k,v in zip(self.s['sinks']['node'],self.s['sinks']['k'],self.s['sinks']['backlog'])}
        value={p:pi.get(p,0) for p in self.pairs}
        for p in self.fuel: value[p]=max(value[p],15000)
        for _ in range(P):
            old=value.copy()
            for a,b,k,es in self.routes: value[a,k]=max(value[a,k],.95*old[b,k])
            for n,ik,ok,typ,d in self.prod: value[n,ik]=max(value[n,ik],.85*old[n,ok])
            if old==value: break
        c=[]; bounds=[]; bal=[[{} for _ in range(P)] for _ in range(H)]; rr=[]; cc=[]; vv=[]; rhs=[]
        def var(price,upper=None):
            j=len(c); c.append(float(price)); bounds.append((0,upper)); return j
        def term(h,p,j,v): bal[h][p][j]=bal[h][p].get(j,0)+v
        def ub(items,b):
            r=len(rhs); rhs.append(float(b))
            for j,v in items: rr.append(r); cc.append(j); vv.append(v)
        inv=np.empty((H,P),int)
        for h in range(H):
            for i in range(P):
                j=var(.02); inv[h,i]=j; term(h,i,j,1)
                if h: term(h,i,inv[h-1,i],-1)
        pending={}
        for j in np.flatnonzero(o.get('pending_prohibitions.edge.observed',[])):
            p=(int(o['pending_prohibitions.edge'][j]),int(o['pending_prohibitions.k'][j]))
            pending[p]=min(pending.get(p,self.T+1),int(o['pending_prohibitions.effective_week'][j]))
        resources={}; flows=np.full((H,A),-1,int); current=[[] for _ in range(P)]
        mask=o.get('action_mask',np.ones(A))
        for h in range(H):
            discount=.995**h
            for a,(src,dst,k,es) in enumerate(self.routes):
                delay=sum(max(1,int(tau[e])) for e in es)
                if h+delay>=H or not mask[a]: continue
                if any(pending.get((e,k),self.T+1)<=week+h for e in es): continue
                cps=set(self.tail[e] for e in es if self.tail[e] in self.cp)
                if any(opened[self.cp[n]]<=0 and week+h<ends.get(n,self.T+1) for n in cps): continue
                price=sum(cost[e]+tariff[e,k]*self.s['commodities']['v'][k] for e in es)
                j=var(price*discount,max(0,min(cap[e] for e in es))); flows[h,a]=j
                term(h,self.idx[src,k],j,1); term(h+delay,self.idx[dst,k],j,-1)
                if h==0: current[self.idx[src,k]].append(j)
                for e in es: resources.setdefault((h,'edge',e),[]).append(j)
                for n in cps: resources.setdefault((h,self.s['commodities']['pool'][k],n),[]).append(j)
            groups={}
            for n,ik,ok,typ,d in self.prod:
                if h+d>=H: continue
                j=var(.1); term(h,self.idx[n,ik],j,1); term(h+d,self.idx[n,ok],j,-1); groups.setdefault((n,typ),[]).append(j)
            for (n,typ),js in groups.items():
                ns=self.l['fabs' if typ=='fab' else 'osats']; key='graph_now.fab.cap_eff' if typ=='fab' else 'graph_now.osat.thr_eff'
                ub([(j,1) for j in js],max(0,read(key,np.zeros(len(ns)))[ns.index(n)]))
            for p,rate in self.fuel.items():
                j=var(-15000*discount,max(0,rate)); term(h,self.idx[p],j,1)
        for i,js in enumerate(current):
            if js: ub([(j,1) for j in js],stock[i]+supply[i])
        for (h,typ,n),js in resources.items():
            limit=cap[n] if typ=='edge' else read('graph_now.kappa.'+typ,np.full(len(self.cp),1e12))[self.cp[n]]*(opened[self.cp[n]] if opened[self.cp[n]]>0 else 1)
            ub([(j,1) for j in js],max(0,limit))
        backlog=read('backlog.qty',np.zeros(len(self.l['demands'])))
        for d,pp in enumerate(self.l['demands']):
            p=tuple(pp); prev=None
            for h in range(H):
                q=max(0,float(forecast[d,min(h,forecast.shape[1]-1)]))
                if back.get(p,False):
                    served=var(0); unserved=var(pi.get(p,10000)*(.995**h))
                    items=[(served,1),(unserved,1)]+([] if prev is None else [(prev,-1)])
                    b=q+(backlog[d] if h==0 else 0)
                    ub(items,b); ub([(j,-v) for j,v in items],-b); prev=unserved
                else: served=var(-pi.get(p,10000)*(.995**h),q)
                term(h,self.idx[p],served,1)
        if week+H-1<self.T:
            for p in self.pairs:
                target=0
                for d,dp in enumerate(self.l['demands']):
                    if p==tuple(dp): target=1.5*float(np.mean(forecast[d]))
                if p in self.fuel: target=2*self.fuel[p]
                if target>0:
                    j=var(-.35*value[p],target); ub([(j,1),(inv[-1,self.idx[p]],-1)],0)
        er=[]; ec=[]; ev=[]; beq=[]
        for h in range(H):
            for i in range(P):
                r=len(beq); beq.append(inc[h,i]+(stock[i] if h==0 else 0))
                for j,v in bal[h][i].items(): er.append(r); ec.append(j); ev.append(v)
        N=len(c)
        res=linprog(c,A_ub=coo_matrix((vv,(rr,cc)),shape=(len(rhs),N)).tocsr(),b_ub=rhs,A_eq=coo_matrix((ev,(er,ec)),shape=(len(beq),N)).tocsr(),b_eq=beq,bounds=bounds,method='highs',options={'time_limit':12})
        out=np.zeros(A); outgoing=np.zeros(P)
        if res.x is not None:
            for a,j in enumerate(flows[0]):
                if j>=0: out[a]=max(0,res.x[j])
        for a,(src,dst,k,es) in enumerate(self.routes): outgoing[self.idx[src,k]]+=out[a]
        self.prev=(stock.copy(),direct.copy(),outgoing)
        return {'flows':out}
