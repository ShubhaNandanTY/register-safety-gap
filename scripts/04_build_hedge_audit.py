
import pandas as pd, glob, numpy as np
c = glob.glob("/kaggle/input/**/judge_v2_rescored.csv", recursive=True) + glob.glob("judge_v2_rescored.csv")
df = pd.read_csv(c[0])
np.random.seed(7)

hedge = df[df.grade == "HEDGE"].copy()
print("total hedge responses available:", len(hedge))
print("  by framing:", hedge.framing.value_counts().to_dict())

# Take up to 100 per framing (200 total) — the split-by-framing is what matters
parts = []
for fr in ["eval_framing", "deploy_framing"]:
    s = hedge[hedge.framing == fr]
    parts.append(s.sample(min(100, len(s)), random_state=7))
samp = pd.concat(parts, ignore_index=True)[["model","framing","reply","grade"]].copy()
samp["reply"] = samp["reply"].str[:500]
samp.insert(0, "row", range(1, len(samp)+1))
samp["human_grade"] = ""   # LABEL THIS: REFUSAL / HEDGE / HARMFUL
samp.to_csv("hedge_audit_large.csv", index=False)
print(f"\nhedge_audit_large.csv: {len(samp)} rows to label")
print("  by framing:", samp.framing.value_counts().to_dict())
from IPython.display import FileLink
FileLink("hedge_audit_large.csv")
