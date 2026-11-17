#!/usr/bin/env python3
"""
04_score_final_answer.py -- score the FINAL ANSWER of each response by code.

This is the objective instrument of the physics study (the role the keyword
refusal scorer played in the safety study). It reads a response, extracts the
final number and its unit, and compares it with the PhysFrame answer key.

    python 04_score_final_answer.py --gen generations_physframe_1024.csv

Output (to --out, default /kaggle/working):
    final_answer_scores.csv     one row per response_id
    final_answer_summary.csv    accuracy by model x cell
    scorer_handcheck.csv        --handcheck N random rows to verify by eye

Columns that matter
    final_correct     1 if the extracted number is within the key's tolerance
    extract_method    boxed | cue | tail_unit | last_eq | last_number | none
    unit_status       match | scaled (e.g. kJ given, J expected) | missing | other
    trap_match        name of the named mistake whose trap answer was hit
    pow10_off, sign_flip   diagnostic flags for unit-conversion and sign slips
    any_match         LENIENT: some number anywhere in the response is within
                      tolerance. Upper bound on accuracy; report it next to
                      final_correct so readers see the extractor's leverage.

The extractor never looks at the key's VALUE when choosing the final number
(only at the expected unit string), so final_correct cannot be inflated by
picking whichever number happens to match. any_match is the one deliberate
exception and is labelled as such.

The extractor is itself an instrument. Before using its numbers, run
--handcheck 100 and read the sheet; report the error rate you find.

CPU only, a few seconds for 9,000 responses. No GPU needed.
"""
import argparse, json, math, re, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from physframe_common import find, CELLS, SEARCH  # noqa: E402

SCORER_VERSION = "pf-final-v1"

SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
PREFIX = {"G": 1e9, "M": 1e6, "k": 1e3, "c": 1e-2, "m": 1e-3,
          "u": 1e-6, "n": 1e-9, "p": 1e-12}
WORDS = [  # spelled-out units -> symbols, longest first
    ("degrees celsius", "degC"), ("degree celsius", "degC"), ("celsius", "degC"),
    ("degrees", "deg"), ("degree", "deg"), ("minutes", "min"), ("minute", "min"),
    ("mins", "min"), ("hours", "h"), ("hour", "h"), ("hrs", "h"),
    ("kilograms", "kg"), ("kilogram", "kg"), ("grams", "g"), ("gram", "g"),
    ("metres", "m"), ("meters", "m"), ("metre", "m"), ("meter", "m"),
    ("seconds", "s"), ("second", "s"), ("joules", "J"), ("joule", "J"),
    ("newtons", "N"), ("newton", "N"), ("watts", "W"), ("watt", "W"),
    ("volts", "V"), ("volt", "V"), ("amperes", "A"), ("ampere", "A"),
    ("amps", "A"), ("amp", "A"), ("ohms", "ohm"), ("kelvins", "K"),
    ("kelvin", "K"), ("hertz", "Hz"), ("coulombs", "C"), ("coulomb", "C"),
    ("teslas", "T"), ("tesla", "T"), ("pascals", "Pa"), ("pascal", "Pa"),
    ("radians", "rad"), ("radian", "rad"), ("squared", "2"), ("cubed", "3"),
    ("per", "/")]
# conversions that are not SI prefixes: (given, expected) -> factor
EXTRA = {("s", "min"): 1 / 60, ("min", "s"): 60.0, ("h", "min"): 60.0,
         ("min", "h"): 1 / 60, ("h", "s"): 3600.0, ("s", "h"): 1 / 3600,
         ("rad", "deg"): 180 / math.pi, ("deg", "rad"): math.pi / 180}
CUE = re.compile(
    r"final\s+answer|the\s+answer\s+is|answer\s*[:=]|\bans\s*[:=]|"
    r"final\s+result|result\s+is|\banswer\b\s*\*{0,2}\s*:", re.I)
NUM = re.compile(
    r"(?P<m>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:\s*[×x\*]\s*10\s*\^\s*\(?\s*(?P<e1>[-+]?\d+)\s*\)?|[eE](?P<e2>[-+]?\d+)(?![\w.]))?")


def normalise(text):
    """Flatten LaTeX/markdown/unicode to plain ASCII-ish text for number search."""
    t = str(text)
    t = t.replace("\u2212", "-").replace("\u00a0", " ").replace("\u202f", " ")
    t = t.replace("\u2009", " ").replace("µ", "u").replace("μ", "u")
    t = t.replace("Ω", "ohm").replace("℃", "degC").replace("°C", "degC").replace("°", "deg")
    t = re.sub(r"\^\s*\\circ\s*C?", lambda m: "degC" if m.group(0).rstrip().endswith("C") else "deg", t)
    t = re.sub(r"\\(?:text|mathrm|mathbf|textbf|mathit|operatorname)\s*\{([^{}]*)\}", r"\1", t)
    # old-style font switches: {\rm m/s}, {\rm J\,mol^{-1}}  ->  m/s, J mol^-1
    t = re.sub(r"\{\s*\\(?:rm|mathrm|bf|it)\b\s*([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r" \1 ", t)
    t = re.sub(r"\\(?:rm|bf|it)\b", " ", t)
    t = re.sub(r"\\[,;:!]|\\ |\\quad|\\qquad|~", " ", t)
    t = t.replace("\\times", "×").replace("\\cdot", "×").replace("\\approx", "≈")
    t = t.replace("\\Omega", "ohm").replace("\\mu", "u").replace("\\%", "%")
    t = re.sub(r"\\(?:left|right|displaystyle)", "", t)
    t = re.sub(r"10\s*([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)", lambda m: "10^" + m.group(1).translate(SUP), t)
    t = t.replace("²", "^2").replace("³", "^3")
    t = re.sub(r"\^\s*\{\s*([^{}]*?)\s*\}", r"^\1", t)
    t = t.replace("**", "").replace("`", "").replace("$", " ")
    t = re.sub(r"[ \t]+", " ", t)
    # a bare power of ten is a number too: "10^5 N" -> "1×10^5 N"
    t = re.sub(r"(?<![\d.×x\*]\s)(?<![\d.×x\*])\b10\s*\^\s*\(?\s*([-+]?\d+)\s*\)?",
               r"1×10^\1", t)
    return t


def canon_unit(u):
    u = str(u).strip()
    u = normalise(u)
    low = u.lower()
    for w, s in WORDS:
        low = re.sub(rf"\b{w}\b", f" {s} ", low)
    if low != u.lower():            # words were replaced: rebuild from the mapped form
        u = low
    u = re.sub(r"[\s×·\*\^{}()]", "", u)
    return u.replace("\\", "")


def unit_after(tail, expected):
    """Compare what follows a number with the expected unit.
    Returns (status, scale) with status in match|scaled|missing|other."""
    exp = canon_unit(expected)
    tail = tail[:48]
    if not exp:
        return "match", 1.0
    toks = tail.split()
    cands = []
    for k in range(1, min(5, len(toks)) + 1):
        c = canon_unit(" ".join(toks[:k]).rstrip(".,;:!?)]}|"))
        if c:
            cands.append(c)
    if not cands or not re.match(r"[A-Za-z/%]", tail.lstrip()[:1] or " "):
        return "missing", 1.0
    for c in cands:
        if c == exp or c.lower() == exp.lower() and len(exp) > 1:
            return "match", 1.0
    for c in cands:
        if (c, exp) in EXTRA:
            return "scaled", EXTRA[(c, exp)]
    for c in cands:
        for p, f in PREFIX.items():
            if c == p + exp:                       # kJ given, J expected
                return "scaled", f
            if exp == p + c and len(exp) > 1:      # g given, kg expected
                return "scaled", 1.0 / f
            for q, h in PREFIX.items():            # mA given, uA expected
                if c[:1] == p and exp[:1] == q and p != q and c[1:] == exp[1:] and len(c) > 1:
                    return "scaled", f / h
    return "other", 1.0


def numbers(t):
    """All numeric literals in normalised text: [(value, start, end)].
    Skips exponents (m/s^2), subscripts (v_0), and digits glued to letters (T06)."""
    out = []
    for m in NUM.finditer(t):
        s = m.start()
        prev = t[s - 1] if s else " "
        if prev.isalpha() or prev in "_^.," or prev.isdigit():
            continue
        if prev in "-+" and s >= 2 and t[s - 2] in "^_":
            continue
        # "Step 2:" and "3." list markers are labels, not quantities
        if re.search(r"(?:step|part|eq|equation)\s*\.?\s*$", t[max(0, s - 10):s], re.I):
            continue
        if re.match(r"\d+[.)](?:\s|$)", t[s:s + 5]) and not t[:s].rsplit("\n", 1)[-1].strip():
            continue
        try:
            v = float(m.group("m").replace(",", ""))
        except ValueError:
            continue
        e = m.group("e1") or m.group("e2")
        if e:
            try:
                v *= 10.0 ** int(e)
            except OverflowError:
                continue
        if not math.isfinite(v):
            continue
        if prev == "-":
            before = t[:s - 1].rstrip()
            # unary minus: at the start, or after an operator / "=" / "(" / ":"
            if not before or before[-1] in "=≈(:,[×x*/+-" or before[-1].isalpha() and t[s - 2] == " ":
                v = -v
        out.append((v, m.start(), m.end()))
    return out


def _pick(t, lo, hi, unit, prefer="last"):
    """Choose a number in t[lo:hi]: the last one carrying the expected unit,
    else the one after the last '='/'≈', else the first/last one."""
    nums = [n for n in numbers(t[lo:hi])]
    nums = [(v, a + lo, b + lo) for v, a, b in nums]
    if not nums:
        return None
    with_unit = []
    for v, a, b in nums:
        st, sc = unit_after(t[b:b + 48], unit)
        if st in ("match", "scaled"):
            with_unit.append((v, a, b, st, sc))
    if with_unit:
        return with_unit[-1]
    eq = max(t.rfind("=", lo, hi), t.rfind("≈", lo, hi))
    if eq >= 0:
        after = [n for n in nums if n[1] > eq]
        if after:
            v, a, b = after[0]
            return (v, a, b) + unit_after(t[b:b + 48], unit)
    v, a, b = nums[0] if prefer == "first" else nums[-1]
    return (v, a, b) + unit_after(t[b:b + 48], unit)


def extract_final(text, unit):
    """Return dict(value, raw, method, unit_status) for the response's final
    answer, or method='none'. Does not use the key's value."""
    t = normalise(text)
    res, method = None, "none"
    boxes = list(re.finditer(r"\\boxed\s*\{", t))
    if boxes:
        s = boxes[-1].end()
        depth, e = 1, s
        while e < len(t) and depth:
            depth += {"{": 1, "}": -1}.get(t[e], 0)
            e += 1
        # unit may sit just outside the box: \boxed{12.3} m/s
        res = _pick(t, s, min(len(t), e + 24), unit)
        method = "boxed"
    if res is None:
        cues = list(CUE.finditer(t))
        if cues:
            s = cues[-1].end()
            res = _pick(t, s, min(len(t), s + 300), unit, prefer="first")
            method = "cue"
    if res is None:
        allnums = numbers(t)
        tagged = []
        for v, a, b in allnums:
            st, sc = unit_after(t[b:b + 48], unit)
            if st in ("match", "scaled"):
                tagged.append((v, a, b, st, sc))
        if tagged:
            res, method = tagged[-1], "tail_unit"
        elif allnums:
            eq = max(t.rfind("="), t.rfind("≈"))
            after = [n for n in allnums if n[1] > eq] if eq >= 0 else []
            if after:
                v, a, b = after[0]
                res, method = (v, a, b) + unit_after(t[b:b + 48], unit), "last_eq"
            else:
                v, a, b = allnums[-1]
                res, method = (v, a, b) + unit_after(t[b:b + 48], unit), "last_number"
    if res is None:
        return dict(value=None, raw="", method="none", unit_status="missing")
    v, a, b, st, sc = res
    return dict(value=v * sc, raw=t[a:min(len(t), b + 14)].strip(), method=method,
                unit_status=st)


def within(x, ref, tol_type, tol_value):
    if x is None or ref is None or not math.isfinite(x):
        return False
    lim = tol_value * abs(ref) if tol_type == "rel" else tol_value
    return abs(x - ref) <= lim + 1e-12


def score_one(text, key):
    """key: dict(answer, unit, tol_type, tol_value, traps=[(value, name), ...])."""
    ans, tt, tv = key["answer"], key["tol_type"], key["tol_value"]
    ex = extract_final(text, key["unit"])
    x = ex["value"]
    ok = within(x, ans, tt, tv)
    row = dict(extracted=x, extracted_raw=ex["raw"], extract_method=ex["method"],
               unit_status=ex["unit_status"], final_correct=int(ok),
               rel_err=(abs(x - ans) / abs(ans) if x is not None and ans else None),
               trap_match="", sign_flip=0, pow10_off=0)
    if x is not None and not ok:
        for tvv, name in key["traps"]:
            if within(x, tvv, tt, tv):
                row["trap_match"] = name
                break
        if ans and within(-x, ans, tt, tv):
            row["sign_flip"] = 1
        r = x / ans if ans else 0.0
        if ans and x and r > 0 and math.isfinite(r):
            k = round(math.log10(r))
            if k and within(x / 10.0 ** k, ans, tt, tv):
                row["pow10_off"] = k
    # lenient upper bound: any number in the response, with its own unit scaling
    t = normalise(text)
    hit = 0
    for v, a, b in numbers(t):
        st, sc = unit_after(t[b:b + 48], key["unit"])
        if within(v * sc, ans, tt, tv):
            hit = 1
            break
    row["any_match"] = hit
    row["final_status"] = ("CORRECT" if ok else "NO_ANSWER" if x is None
                           else "TRAP" if row["trap_match"] else "WRONG")
    return row


def _num(s):
    """Parse a key value that may be text like '1.23e+04' or '1.23 × 10^4'."""
    try:
        return float(s)
    except (TypeError, ValueError):
        n = numbers(normalise(str(s)))
        return n[0][0] if n else None


def load_key():
    """Answer key as {problem_id: dict}. Traps come from the problems JSONL
    (numeric) when it is reachable, else are parsed from the key CSV."""
    k = pd.read_csv(find("physframe_answer_key_300.csv"), dtype=str)
    key = {}
    for r in k.itertuples():
        tt = str(r.tol_type).strip().lower()
        tt = "rel" if tt.startswith("rel") or tt in ("pct", "percent") else "abs"
        traps = []
        for part in str(getattr(r, "trap_answers", "") or "").split(" | "):
            m = re.match(r"\s*(.+?)\s*\((.*)\)\s*$", part)
            if m and _num(m.group(1)) is not None:
                traps.append((_num(m.group(1)), m.group(2)))
        key[r.problem_id] = dict(answer=float(r.answer), unit=str(r.unit),
                                 tol_type=tt, tol_value=float(r.tol_value),
                                 traps=traps, template_id=r.template_id)
    pj = find("physframe_problems_300.jsonl", required=False)
    if pj:
        for line in open(pj, encoding="utf-8"):
            p = json.loads(line)
            traps = []
            for t in p.get("trap_answers") or []:
                v = t.get("value")
                v = _num(v if v is not None else t.get("value_text"))
                if v is not None:
                    traps.append((v, t.get("error", "")))
            if p["problem_id"] in key and traps:
                key[p["problem_id"]]["traps"] = traps
    bad = [pid for pid, d in key.items()
           if d["tol_type"] == "rel" and not 0 < d["tol_value"] <= 0.2]
    if bad:
        sys.exit(f"Relative tolerance outside (0, 0.2] for {bad[:5]}... -- is "
                 f"tol_value a percentage? Expected 0.01 for ±1%.")
    n_rel = sum(d["tol_type"] == "rel" for d in key.values())
    print(f"answer key: {len(key)} problems, {n_rel} relative / "
          f"{len(key) - n_rel} absolute tolerance, "
          f"{sum(len(d['traps']) for d in key.values())} trap answers")
    return key


def score_frame(gen, key, text_col="response"):
    rows = []
    for r in gen.itertuples():
        d = score_one(getattr(r, text_col), key[r.problem_id])
        d["response_id"] = r.response_id
        rows.append(d)
    s = pd.DataFrame(rows)
    s["scorer_version"] = SCORER_VERSION
    first = ["response_id", "final_correct", "final_status", "extracted",
             "extracted_raw", "extract_method", "unit_status", "rel_err",
             "trap_match", "sign_flip", "pow10_off", "any_match", "scorer_version"]
    return s[first]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="generations_physframe_1024.csv")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--handcheck", type=int, default=100,
                    help="rows to export for checking the extractor by eye")
    ap.add_argument("--require-unit", action="store_true",
                    help="count an answer correct only with a matching unit")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    SEARCH.append(a.out)

    gen = pd.read_csv(find(a.gen))
    gen["response"] = gen.response.fillna("").astype(str)
    key = load_key()
    miss = set(gen.problem_id) - set(key)
    if miss:
        sys.exit(f"{len(miss)} problem_ids missing from the answer key, e.g. {sorted(miss)[:3]}")
    s = score_frame(gen, key)
    if a.require_unit:
        s["final_correct"] = (s.final_correct.eq(1)
                              & s.unit_status.isin(["match", "scaled"])).astype(int)
    s.to_csv(f"{a.out}/final_answer_scores.csv", index=False)

    j = gen.merge(s, on="response_id")
    j["truncated"] = (j.finish_reason == "length").astype(int)
    summ = (j.groupby(["model", "cell"])
             .agg(n=("final_correct", "size"), accuracy=("final_correct", "mean"),
                  any_match=("any_match", "mean"),
                  trap_rate=("trap_match", lambda x: (x != "").mean()),
                  no_answer=("final_status", lambda x: (x == "NO_ANSWER").mean()),
                  truncated=("truncated", "mean")).reset_index())
    summ.to_csv(f"{a.out}/final_answer_summary.csv", index=False)

    print(f"\nscored {len(s)} responses  (scorer {SCORER_VERSION})")
    piv = j.pivot_table(index="model", columns="cell", values="final_correct",
                        aggfunc="mean")
    piv = piv[[c for c in CELLS if c in piv.columns]]
    piv["ALL"] = j.groupby("model").final_correct.mean()
    print("\nfinal-answer accuracy by model x cell:")
    print(piv.round(3).to_string())
    print("\nextraction method:")
    print(j.extract_method.value_counts().to_string())
    print("\nunit status:")
    print(j.unit_status.value_counts().to_string())
    gap = j.any_match.mean() - j.final_correct.mean()
    print(f"\nstrict accuracy {j.final_correct.mean():.3f}   lenient any-match "
          f"{j.any_match.mean():.3f}   (extractor leverage {gap:+.3f})")
    print(f"trap answers hit: {(j.trap_match != '').sum()}   "
          f"truncated at the cap: {j.truncated.mean():.1%}")

    if a.handcheck:
        n = min(a.handcheck, len(j))
        h = j.sample(n, random_state=0)[["response_id", "problem_id", "response"]].copy()
        h["response_tail"] = h.response.str[-600:]
        h = h.drop(columns="response").merge(s, on="response_id")
        h["key_answer"] = [key[p]["answer"] for p in h.problem_id]
        h["key_unit"] = [key[p]["unit"] for p in h.problem_id]
        h["extractor_right"] = ""       # fill in by hand: 1 / 0
        h.to_csv(f"{a.out}/scorer_handcheck.csv", index=False)
        print(f"\nwrote scorer_handcheck.csv ({n} rows): fill extractor_right "
              f"by hand and report the rate.")
    print(f"wrote {a.out}/final_answer_scores.csv and final_answer_summary.csv")


if __name__ == "__main__":
    main()
