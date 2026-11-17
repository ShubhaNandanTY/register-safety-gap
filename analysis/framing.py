import pandas as pd, numpy as np
rng=np.random.default_rng(1)
d=pd.read_csv("responses.csv"); d["jp"]=d.judge_label.isin(["CORRECT","RIGHT_ANSWER_WRONG_REASONING"]).astype(int); d["ag"]=(d.jp==d.code_correct).astype(int)
FORMAL,CASUAL=["bench","formal_motive"],["casual","casual_motive"]
def reg(g,col): 
    c=g.groupby("framing")[col].mean()*100
    return (c.get("casual",np.nan)+c.get("casual_motive",np.nan)-c.get("bench",np.nan)-c.get("formal_motive",np.nan))/2
def mot(g,col):
    c=g.groupby("framing")[col].mean()*100
    return (c["formal_motive"]+c["casual_motive"]-c["bench"]-c["casual"])/2
def plc(g,col):
    c=g.groupby("framing")[col].mean()*100; return c["placebo"]-c["bench"]
def boot(g,f,unit="template",B=1000):
    gs=[h for _,h in g.groupby(unit)]; v=[]
    for _ in range(B):
        v.append(f(pd.concat([gs[i] for i in rng.integers(0,len(gs),len(gs))])))
    return np.percentile(v,[2.5,97.5]), np.std(v)
rows=[]
order=["Qwen3-Omni-30B","Qwen2.5-7B","Qwen2.5-3B","Gemma-2-9B","Qwen2.5-1.5B","Gemma-2-2B","Qwen2.5-0.5B"]
top4=order[:4]
sets={**{m:d[d.model==m] for m in order},"pooled top-4":d[d.model.isin(top4)],"pooled all 7":d}
for name,g in sets.items():
    r={"set":name}
    for col,lab in [("ag","agree"),("code_correct","acc")]:
        for fn,fl in [(reg,"register"),(mot,"motive"),(plc,"placebo")]:
            est=fn(g,col); (lo,hi),sd=boot(g,lambda h:fn(h,col)); (plo,phi),psd=boot(g,lambda h:fn(h,col),unit="problem",B=500)
            r[f"{lab}_{fl}"]=f"{est:+.1f} [{lo:+.1f},{hi:+.1f}] p[{plo:+.1f},{phi:+.1f}]"
            r[f"{lab}_{fl}_sdT"]=round(sd,2); r[f"{lab}_{fl}_sdP"]=round(psd,2)
    rows.append(r); print(r["set"], {k:v for k,v in r.items() if not k.endswith(("sdT","sdP")) and k!="set"}, flush=True)
pd.DataFrame(rows).to_csv("framing_table.csv",index=False)
