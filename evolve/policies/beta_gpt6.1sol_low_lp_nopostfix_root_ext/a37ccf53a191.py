# 0.6849165903319708
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


class Agent:
    def __init__(self, config):
        self.c = config
        self.s = config['static']; self.l = config['layout']
        self.raw = self.s['instance']; self.nodes = self.raw['nodes']
        self.kid = {v:i for i,v in enumerate(self.s['commodities']['id'])}
        self.slots = list(map(tuple,self.l['stock_slots']))
        self.si = {v:i for i,v in enumerate(self.slots)}
        self.cp = {v:i for i,v in enumerate(self.l['chokepoints'])}
        self.fi = {v:i for i,v in enumerate(self.l['fabs'])}
        self.gi = {v:i for i,v in enumerate(self.l['grids'])}
        self.oi = {v:i for i,v in enumerate(self.l['osats'])}
        self.cache = {}
        self.storage=[]; self.holding=[]; self.salvage=[]
        for n,k in self.slots:
            d=self.nodes[n].get('stock',{}).get(self.s['commodities']['id'][k],{})
            self.storage.append(float(d['storage']) if d.get('storage') is not None else 1e12)
            self.holding.append(float(d.get('holding_cost',0)))
            self.salvage.append(float(d.get('salvage',0)))
        self.routes=[]; self.dup=[]
        for j,e in enumerate(self.s['action_slots']['edge']):
            ln=self.s['action_slots']['lane'][j]
            es=list(self.s['lanes']['edges'][ln]) if ln is not None and ln>=0 else [e]
            self.routes.append((e,self.s['action_slots']['k'][j],ln,es,self.s['edges']['tail'][e],self.s['edges']['head'][es[-1]]))
            alt=self.s['lanes']['alt_of'][ln] if ln is not None and ln>=0 else self.s['edges']['alt_of'][e]
            dt=0
            if alt and self.s['edges']['mode'][e]=='sea':
                base=list(self.s['lanes']['edges'][alt['lane']]) if 'lane' in alt else [alt['edge']]
                dt=max(0,sum(self.s['edges']['tau0'][x] for x in es)-sum(self.s['edges']['tau0'][x] for x in base))
            self.dup.append(dt)

    def act(self,o):
        def get(key,default):
            a=np.asarray(o.get(key,default),dtype=float); m=o.get(key+'.observed')
            if m is not None and np.shape(m)==a.shape:
                a=np.where(m,a,self.cache.get(key,np.asarray(default,dtype=float)))
            self.cache[key]=a.copy(); return a
        week=int(o['week'][0]); remaining=int(self.c['T'])-week+1
        H=min(26,remaining); S=len(self.slots); E=len(self.s['edges']['id']); K=len(self.kid)
        stock=get('stock.qty',np.zeros(S))
        tau=np.maximum(1,get('graph_now.tau',self.s['edges']['tau0'])).astype(int)
        freight=get('graph_now.c',self.s['edges']['c0'])
        caps=get('graph_now.u',[0 if x is None else x for x in self.s['edges']['u0']])
        prohibited=get('graph_now.prohibited',np.zeros((E,K)))
        tariff=get('graph_now.tariff',np.zeros((E,K)))
        opening=get('graph_now.open',np.ones(len(self.cp)))
        war=get('graph_now.war_risk',np.zeros(len(self.cp))).astype(int)
        kap={p:get('graph_now.kappa.'+p,np.full(len(self.cp),1e12)) for p in ('tb','ct')}
        ends={}; pending={}
        for z in np.flatnonzero(o.get('closure_end.chokepoint.observed',[])):
            if o.get('closure_end.end_week.observed',np.zeros(len(o['closure_end.chokepoint'])))[z]:
                n=int(o['closure_end.chokepoint'][z]); ends[n]=max(ends.get(n,week),int(o['closure_end.end_week'][z]))
        for z in np.flatnonzero(o.get('pending_prohibitions.edge.observed',[])):
            key=(int(o['pending_prohibitions.edge'][z]),int(o['pending_prohibitions.k'][z]))
            pending[key]=min(pending.get(key,10**9),int(o['pending_prohibitions.effective_week'][z]))
        def recovery(n): return max(0,ends.get(n,week+4)-week+1)
        def suffix(e,ln):
            if ln is not None and ln>=0:
                es=self.s['lanes']['edges'][int(ln)]
                if e in es: return list(es[es.index(e)+1:])
            return []
        def poolcap(n,p,h):
            ci=self.cp[n]; cap=kap[p][ci]
            if opening[ci]<.1 and h>=recovery(n):
                d=self.nodes[n].get('chokepoint',{}); mu=d.get('mu',{})
                if isinstance(mu,dict): cap=float(d.get('k_c',1))*float(mu.get(p,cap))
            return max(0,cap)
        def path(es,start):
            t=start; visits=[]
            for e in es:
                n=self.s['edges']['tail'][e]
                if n in self.cp and opening[self.cp[n]]<.1: t=max(t,recovery(n))
                visits.append((e,t)); t+=int(tau[e])
            return t,visits
        arrivals=np.zeros((S,H)); occupied_e={}; occupied_p={}
        def arrive(n,k,q,h):
            s=self.si.get((int(n),int(k)))
            if s is not None and 0<=h<H: arrivals[s,int(h)]+=q
        def schedule(es,k,q,start):
            if q<=1e-9 or start>=H or not es: return
            e=es[0]; n=self.s['edges']['tail'][e]
            if n not in self.cp:
                finish=start+int(tau[e])
                if len(es)==1: arrive(self.s['edges']['head'][e],k,q,finish)
                else: schedule(es[1:],k,q,finish)
                return
            p=self.s['commodities']['pool'][k]; h=max(0,int(start))
            if opening[self.cp[n]]<.1: h=max(h,recovery(n))
            if prohibited[e,k]: return
            while q>1e-9 and h<H:
                if week+h>=pending.get((e,k),10**9): break
                take=min(q,max(0,caps[e]-occupied_e.get((e,h),0)),max(0,poolcap(n,p,h)-occupied_p.get((n,p,h),0)))
                if take>1e-9:
                    occupied_e[e,h]=occupied_e.get((e,h),0)+take
                    occupied_p[n,p,h]=occupied_p.get((n,p,h),0)+take
                    finish=h+int(tau[e])
                    if len(es)==1: arrive(self.s['edges']['head'][e],k,take,finish)
                    else: schedule(es[1:],k,take,finish)
                    q-=take
                h+=1
        jobs=[]
        for z,(n,k,ln,e) in enumerate(self.l.get('lot_keys',[])):
            if 'queue_lots.qty' in o:
                for cohort in np.flatnonzero(o['queue_lots.qty'][z]>0):
                    jobs.append((int(cohort)-int(self.c['T']),0,k,float(o['queue_lots.qty'][z,cohort]),[e]+suffix(e,ln)))
        pq=o.get('pipeline.qty',[])
        for z in np.flatnonzero(o.get('pipeline.qty.observed',np.asarray(pq)>0)):
            e=int(o['pipeline.edge'][z]); k=int(o['pipeline.k'][z])
            ln=int(o['pipeline.lane'][z]) if o.get('pipeline.lane.observed',np.zeros(len(pq)))[z] else None
            start=int(o['pipeline.arrival_week'][z])-week; rest=suffix(e,ln)
            if rest: jobs.append((start,start,k,float(pq[z]),rest))
            else: arrive(self.s['edges']['head'][e],k,pq[z],start)
        for _,start,k,q,es in sorted(jobs,key=lambda x:x[0]): schedule(es,k,q,start)
        for z in np.flatnonzero(o.get('wip.qty.observed',[])):
            arrive(o['wip.node'][z],o['wip.k'][z],o['wip.qty'][z],int(o['wip.out_week'][z])-week)
        obj=[]; bounds=[]; eq=[]; er=[]; ub=[]; br=[]
        def var(c=0,hi=None):
            j=len(obj); obj.append(float(c)/100000); bounds.append((0,hi)); return j
        def row(rows,rhs,terms,b): rows.append(terms); rhs.append(float(b))
        I=[[var(self.holding[s]-(self.salvage[s] if remaining==H and h==H-1 else 0),self.storage[s]) for h in range(H)] for s in range(S)]
        bal=[[{I[s][h]:1} for h in range(H)] for s in range(S)]
        draws=[[{} for h in range(H)] for s in range(S)]
        def term(s,h,x,v):
            if s is not None and 0<=h<H: bal[s][h][x]=bal[s][h].get(x,0)+v
        def buffer(s,h,target,penalty):
            d=var(penalty); row(ub,br,{I[s][h]:-1,d:-1},-max(0,min(target,self.storage[s]*.9)))
        for s,(n,k) in enumerate(self.slots):
            for h in range(H): term(s,h,var(float(self.raw['commodities'][k].get('disposal_cost',3000))),1)
        X=[]; edge_use={}; pool_use={}; fleet_use={}
        mask=o.get('action_mask',np.ones(len(self.routes)))
        for j,(e,k,ln,es,src,dst) in enumerate(self.routes):
            a=self.si.get((src,k)); b=self.si.get((dst,k)); p=self.s['commodities']['pool'][k]
            price=sum(freight[x]+tariff[x,k]*self.s['commodities']['v'][k] for x in es)
            for edge in es:
                n=self.s['edges']['tail'][edge]
                if n in self.cp:
                    charges=self.nodes[n].get('chokepoint',{}).get('war_risk_cost',{}).get(self.s['commodities']['id'][k],{})
                    w=min(2,max(0,war[self.cp[n]]))
                    if isinstance(charges,dict): price+=float(charges.get(('none','red_sea','hormuz_2026')[w],0))
                    elif isinstance(charges,(list,tuple)) and len(charges)>w: price+=float(charges[w])
            xs=[]
            for h in range(H):
                arrival,visits=path(es,h)
                blocked=any(prohibited[x,k] or week+t>=pending.get((x,k),10**9) for x,t in visits)
                hi=max(0,caps[e])
                if blocked or a is None or b is None or (h==0 and not mask[j]) or arrival>=H: hi=0
                x=var(price+.01*(arrival-h),hi); xs.append(x)
                term(a,h,x,1); term(b,arrival,x,-1)
                if a is not None: draws[a][h][x]=1
                if self.dup[j]: fleet_use.setdefault((p,h),{})[x]=self.dup[j]
                for edge,t in visits:
                    if t<H:
                        edge_use.setdefault((edge,t),{})[x]=1
                        n=self.s['edges']['tail'][edge]
                        if n in self.cp: pool_use.setdefault((n,p,t),{})[x]=1
            X.append(xs)
        for (e,h),terms in edge_use.items(): row(ub,br,terms,max(0,caps[e]-occupied_e.get((e,h),0)))
        for (n,p,h),terms in pool_use.items(): row(ub,br,terms,max(0,poolcap(n,p,h)-occupied_p.get((n,p,h),0)))
        params=self.raw.get('params',{})
        fs=params.get('fleet_share'); fm=params.get('fleet_measure')
        if fs is not None and fm is not None:
            for (p,h),terms in fleet_use.items():
                i=('tb','ct').index(p)
                share=fs.get(p,0) if isinstance(fs,dict) else fs[i]
                measure=fm.get(p,0) if isinstance(fm,dict) else fm[i]
                row(ub,br,terms,max(0,float(share)*float(measure)))
        supply=get('graph_now.supply.avail',np.zeros(len(self.l['supply_slots'])))
        for z,key in enumerate(self.l['supply_slots']):
            s=self.si.get(tuple(key))
            if s is not None:
                for h in range(H): term(s,h,var(0,max(0,supply[z])),-1)
        R=get('graph_now.fab.R',np.ones(len(self.fi)))
        alpha=get('graph_now.fab.alpha_bar',np.ones(len(self.fi)))
        G=get('graph_now.grid.G_bar',[self.nodes[n]['grid']['deliverable'] for n in self.gi])
        Y=get('graph_now.grid.y_bar',[self.nodes[n]['grid']['base_load'] for n in self.gi])
        targets={}; fuel_rates={}
        for n,gi in self.gi.items():
            g=self.nodes[n]['grid']; load=float(Y[gi])
            for fn,fi in self.fi.items():
                f=self.nodes[fn]['fab']
                if f.get('grid')==self.s['nodes']['id'][n] and R[fi]>0: load+=float(f['e'])*float(f['cap0'])*alpha[fi]
            fraction=min(1,load/max(1e-12,G[gi]))
            for fuel,share in g['shares'].items():
                if fuel not in self.kid: continue
                k=self.kid[fuel]; s=self.si.get((n,k))
                if s is None: continue
                rate=float(share)*G[gi]*fraction; fuel_rates[s]=rate
                ibar=float(g.get('ibar',{}).get(fuel,0)); targets[s]=max(ibar,rate*5)
                threshold=float(params.get('psi',0))*ibar
                for h in range(H):
                    burn=var(0,rate); lack=var(float(g.get('voll',4100000)),rate)
                    term(s,h,burn,1); row(eq,er,{burn:1,lack:1},rate)
                    if fuel==g.get('rationed') and threshold>0:
                        mx=float(share)*G[gi]/threshold
                        if h==0: row(ub,br,{burn:1},max(0,mx*stock[s]))
                        else: row(ub,br,{burn:1,I[s][h-1]:-mx},0)
                    if remaining-h>1: buffer(s,h,min(max(threshold*1.15 if fuel==g.get('rationed') else 0,rate*3),rate*(remaining-h-1)),14000)
                terminals=set(src for e,kk,ln,es,src,dst in self.routes if kk==k and dst==n and self.nodes[src]['type']=='terminal')
                for tn in terminals:
                    ts=self.si.get((tn,k))
                    if ts is not None:
                        targets[ts]=rate*1.5/max(1,len(terminals))
                        for h in range(H):
                            if remaining-h>3: buffer(ts,h,rate*.75/max(1,len(terminals)),2200)
        for n,fi in self.fi.items():
            f=self.nodes[n]['fab']; a=self.si[n,self.kid[f['input']]]; b=self.si[n,self.kid[f['product']]]
            capacity=max(0,float(f['cap0'])*R[fi]*alpha[fi]); targets[a]=min(self.storage[a]*.8,float(f['cap0'])*2.5)
            for h in range(H):
                x=var(0,capacity); term(a,h,x,1); term(b,h+int(f['tau']),x,-1)
                if remaining-h>int(f['tau'])+5: buffer(a,h,capacity*.9,650)
        thr=get('graph_now.osat.thr_eff',[self.nodes[n]['osat']['thr'] for n in self.oi])
        for n,oi in self.oi.items():
            os=self.nodes[n]['osat']
            for h in range(H):
                terms={}
                for raw,pk in os['packages'].items():
                    a=self.si[n,self.kid[raw]]; b=self.si[n,self.kid[pk]]
                    x=var(); terms[x]=1; term(a,h,x,1); term(b,h+int(os['tau']),x,-1)
                    if len(os['packages'])==1 and remaining-h>int(os['tau'])+3: buffer(a,h,max(0,thr[oi])*.35,500)
                row(ub,br,terms,max(0,thr[oi]))
        forecast=get('demand_forecast.qty',np.zeros((len(self.l['demands']),1)))
        backlog=get('backlog.qty',np.zeros(len(self.l['demands'])))
        for di,(n,k) in enumerate(self.l['demands']):
            s=self.si[n,k]; back=bool(self.s['sinks']['backlog'][di]); prev=None
            penalty=float(self.s['sinks']['pi'][di])
            for h in range(H):
                demand=max(0,forecast[di,min(h,forecast.shape[1]-1)])
                served=var(0,None if back else demand); missing=var(penalty)
                term(s,h,served,1); terms={served:1,missing:1}
                if back and prev is not None: terms[prev]=-1
                row(eq,er,terms,demand+(backlog[di] if back and h==0 else 0)); prev=missing
                if remaining-h>2: buffer(s,h,demand*.85,penalty*.10)
            targets[s]=min(self.storage[s]*.8,np.mean(forecast[di])*1.7)
        for s in range(S):
            for h in range(H):
                if h>0: bal[s][h][I[s][h-1]]=-1
                row(eq,er,bal[s][h],arrivals[s,h]+(stock[s] if h==0 else 0))
                terms=draws[s][h].copy()
                if h>0: terms[I[s][h-1]]=-1
                if terms: row(ub,br,terms,stock[s] if h==0 else 0)
            if remaining>H and s in targets: buffer(s,H-1,targets[s],250000 if s in fuel_rates else 1800)
        def matrix(rows):
            rr=[]; cc=[]; vv=[]
            for i,terms in enumerate(rows):
                for j,v in terms.items():
                    if v: rr.append(i); cc.append(j); vv.append(v)
            return coo_matrix((vv,(rr,cc)),shape=(len(rows),len(obj))).tocsr()
        result=linprog(obj,A_ub=matrix(ub),b_ub=np.asarray(br),A_eq=matrix(eq),b_eq=np.asarray(er),bounds=bounds,method='highs',options={'time_limit':8})
        flows=np.zeros(len(self.routes))
        if result.x is not None:
            for j in range(len(flows)): flows[j]=max(0,float(result.x[X[j][0]]))
        return {'flows':flows}
