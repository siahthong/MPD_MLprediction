#!/usr/bin/env python3
"""
split_dataset.py - CoRE MOF group split (8 : 1 : 1)

[KO] ê°™ì€ MOF ê°€ ë‹¤ë¥¸ coreid ë¡œ ì¤‘ë³µ ìˆ˜ë¡ë˜ì–´ ìžˆì–´, ê·¸ëƒ¥ ë¬´ìž‘ìœ„ë¡œ ë‚˜ëˆ„ë©´
     ê°™ì€ ë¬¼ì§ˆì´ train ê³¼ test ì— ë™ì‹œì— ë“¤ì–´ê°„ë‹¤(data leakage).
     MOFid ë¡œ ì¤‘ë³µì„ ë¬¶ì€ ë’¤ ê·¸ë£¹ ë‹¨ìœ„ë¡œ ë‚˜ëˆˆë‹¤.
[EN] The same MOF appears under different coreids. A plain random split puts
     the same material in both train and test (data leakage). This script
     groups duplicates by MOFid first, then splits at the group level.

--------------------------------------------------------------------
[KO] ë¹ ë¥¸ ì‚¬ìš©ë²• / [EN] Quick start

  pip install pandas numpy scikit-learn

  # [KO] target ì»¬ëŸ¼ì´ CSV ì•ˆì— ìžˆì„ ë•Œ / [EN] target column is in the CSV
  python split_dataset.py --input data.csv --outdir splits \
      --label-cols KH_Label_binary

  # [KO] target ì´ ì•„ì§ ì—†ì„ ë•Œ (êµ¬ì¡°ë§Œìœ¼ë¡œ split ë¨¼ì € ë§Œë“¤ê¸°)
  # [EN] no target yet - build the split from structures only
  python split_dataset.py --input data.csv --outdir splits --no-target

  # [KO] target ì´ ë‹¤ë¥¸ íŒŒì¼ì— ìžˆì„ ë•Œ (multi-target ë„ ê°€ëŠ¥)
  # [EN] targets live in a separate file (multi-target supported)
  python split_dataset.py --input data.csv --outdir splits \
      --target-file targets.csv

--------------------------------------------------------------------
[KO] ìž…ë ¥ CSV ì— ë°˜ë“œì‹œ ìžˆì–´ì•¼ í•˜ëŠ” ì»¬ëŸ¼
[EN] Required columns in the input CSV

  coreid        [KO] êµ¬ì¡° ID            [EN] structure ID
  mofid-v1      [KO] MOFid v1 ë¬¸ìžì—´    [EN] MOFid v1 string
  mofid-v2      [KO] MOFid v2 ë¬¸ìžì—´    [EN] MOFid v2 string
  Metal Types   [KO] ê¸ˆì† ì¢…ë¥˜          [EN] metal types, e.g. "Zn" or "Cu,Si"

  [KO] 'Has OMS' ì»¬ëŸ¼ì´ ìžˆìœ¼ë©´ OMS ë³´ìœ  êµ¬ì¡°ë¥¼ ìžë™ ì œê±°í•œë‹¤.
  [EN] If a 'Has OMS' column exists, OMS-bearing structures are auto-removed.

--------------------------------------------------------------------
[KO] ê·¸ë£¹í‚¤ = ê¸ˆì† | ë§ì»¤ | topology  (ì…‹ì´ ëª¨ë‘ ê°™ì•„ì•¼ ê°™ì€ MOF)
[EN] Group key = metal | linker | topology  (all three must match)

  [KO] ê¸ˆì†     Metal Types ì»¬ëŸ¼ì„ í™•ì •ê°’ìœ¼ë¡œ ì‚¬ìš©. MOFid ì—ì„œ ë½‘ì€ ê¸ˆì†ê³¼
                ëŒ€ì¡°í•´ ê²€ì¦í•œë‹¤. MOFid ê°€ ê¸°ë‘¥ ê¸ˆì†(SiF6/NbOF5/TiF6/ZrF6)ì„
                ë¹ ëœ¨ë¦¬ëŠ” ê²½ìš°ê°€ ìžˆì–´ ì»¬ëŸ¼ ìª½ì„ ì‹ ë¢°í•œë‹¤.
  [EN] metal    The Metal Types column is authoritative. Metals parsed from
                MOFid are used only to validate it. MOFid omits pillar anions
                (SiF6/NbOF5/TiF6/ZrF6), so the column is trusted.

  [KO] ë§ì»¤     MOFid SMILES ì„±ë¶„ ì¤‘ ê¸ˆì†ì´ ì—†ëŠ” ê²ƒ. ì •ë ¬í•´ ìˆœì„œ ë¬´ê´€í•˜ê²Œ.
  [EN] linker   MOFid SMILES components with no metal. Sorted for order-independence.

  [KO] topology MOFid v1 ìš°ì„ , ë¶ˆëª…ì´ë©´ v2. ë‘˜ ë‹¤ ë¶ˆëª…ì´ë©´ ê·¸ë£¹í™”í•˜ì§€ ì•Šê³ 
                ë‹¨ë… ì²˜ë¦¬í•œë‹¤(ëª¨ë¥´ëŠ” ì •ë³´ë¡œ ë¬¶ì§€ ì•ŠëŠ”ë‹¤).
  [EN] topology MOFid v1 first, v2 as fallback. If both are unknown the
                structure is left as a singleton (never group on unknowns).

  [KO] v1 í‚¤ì™€ v2 í‚¤ë¥¼ ëª¨ë‘ ë§Œë“¤ì–´ í•˜ë‚˜ë¼ë„ ì¼ì¹˜í•˜ë©´ ê°™ì€ ê·¸ë£¹ (union-find).
  [EN] Both v1 and v2 keys are built; matching either one merges the group
       (union-find connected components).

[KO] ë¶„í•   StratifiedGroupKFold(10) -> fold 0-7 train / 8 val / 9 test
[EN] Split  StratifiedGroupKFold(10) -> folds 0-7 train / 8 val / 9 test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        if _s is not None and (_s.encoding or "").lower().replace("-", "") != "utf8":
            _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ID_COL = "coreid"
V1_COL, V2_COL = "mofid-v1", "mofid-v2"
METAL_COL = "Metal Types"
REFCODE_COL = "refcode"
OMS_COL = "Has OMS"

# [KO] B, Se, As, Te ëŠ” ë§ì»¤ êµ¬ì„± ì›ì†Œë¡œ ë‚˜íƒ€ë‚˜ë¯€ë¡œ ì œì™¸. Si ëŠ” SiF6 ê¸°ë‘¥ìœ¼ë¡œ ì‹¤ìž¬.
# [EN] B, Se, As, Te appear inside linkers -> excluded. Si is a real SiF6 pillar.
METALS = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc",
    "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Th", "U", "Si",
}
BAD = {"", "nan", "none", "null", "na", "n/a", "unknown", "error", "?", "-"}
_MOFID = re.compile(r"\sMOFid-v[12]\.(\S+)")
_BRACKET = re.compile(r"\[([^\]]+)\]")
_ELEMENT = re.compile(r"([A-Z][a-z]?)")
OMS_MAP = {"yes": True, "no": False, "true": True, "false": False,
           "1": True, "0": False, "y": True, "n": False}


class Log:
    """[KO] í™”ë©´ ì¶œë ¥ê³¼ íŒŒì¼ ê¸°ë¡ì„ ë™ì‹œì— / [EN] print to screen and buffer to file."""

    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        self.lines.append(s)

    def save(self, path: Path):
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


# ============================================================
# [KO] MOFid í•´ì²´  /  [EN] MOFid decomposition
# ============================================================

def parse_mofid(s):
    """[KO] MOFid ë¬¸ìžì—´ -> (ê¸ˆì†ì§‘í•©, ë§ì»¤, topology). ì‹¤íŒ¨ ì‹œ None.
    [EN] MOFid string -> (metal set, linker, topology). None on failure.

    [KO] í˜•ì‹ / [EN] format:
      <SMILES> MOFid-vN.<topology>.<catenation>;<source id>
    """
    if not isinstance(s, str) or s.strip().lower() in BAD:
        return None
    # [KO] ';' ë’¤ ì¶œì²˜ ì‹ë³„ìžëŠ” êµ¬ì¡°ë§ˆë‹¤ ë‹¬ë¼ ë°˜ë“œì‹œ ì œê±°
    # [EN] the part after ';' differs per structure - must be stripped
    body = s.split(";")[0].strip()
    m = _MOFID.search(body)
    if not m:
        return None
    smiles = body[:m.start()].strip()
    if not smiles or smiles.lower() in BAD:
        return None

    metals, linkers = set(), []
    for comp in smiles.split("."):
        if not comp:
            continue
        # [KO] SMILES ê·œê²©ìƒ ê¸ˆì†ì€ ë°˜ë“œì‹œ ëŒ€ê´„í˜¸ ì•ˆì— ì˜¨ë‹¤ -> ìœ„ì¹˜ì™€ ë¬´ê´€í•˜ê²Œ ë¶„ë¦¬
        # [EN] per SMILES spec metals are always bracketed -> position-independent
        found = {e for inner in _BRACKET.findall(comp)
                 for e in _ELEMENT.findall(inner.split("_")[0]) if e in METALS}
        if found:
            metals |= found            # [KO] ê¸ˆì† ë…¸ë“œ / [EN] metal node
        else:
            linkers.append(comp)       # [KO] ë§ì»¤      / [EN] linker

    # [KO] topology í† í° ì¤‘ í•˜ë‚˜ë¼ë„ ë¶ˆëŸ‰ì´ë©´ ì „ì²´ë¥¼ ë¶ˆëª… ì²˜ë¦¬
    # [EN] if any topology token is bad, treat the whole field as unknown
    toks = [t.strip().lower() for t in m.group(1).split(".")[0].split(",") if t.strip()]
    topo = "*" if (not toks or any(t in BAD for t in toks)) else ",".join(sorted(set(toks)))
    return metals, ".".join(sorted(linkers)), topo


class DSU:
    """[KO] Union-Find. êµ¬ì¡°ì™€ í‚¤ë¥¼ ê°™ì€ ê·¸ëž˜í”„ì˜ ë…¸ë“œë¡œ ë‘ê³  ì—°ê²°ì„±ë¶„ì„ ì°¾ëŠ”ë‹¤.
    [EN] Union-Find. Structures and keys are nodes of one graph; groups are
         its connected components."""

    def __init__(self):
        self.p: dict = {}

    def find(self, x):
        self.p.setdefault(x, x)
        r = x
        while self.p[r] != r:
            r = self.p[r]
        while self.p[x] != r:          # [KO] ê²½ë¡œ ì••ì¶• / [EN] path compression
            self.p[x], x = r, self.p[x]
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def mofid_terminal_id(s):
    """[KO] MOFid ë¬¸ìžì—´ì˜ ';' ë’¤ ì¢…ë‹¨ ì‹ë³„ìž. ì—†ìœ¼ë©´ None.
    [EN] the identifier after ';' in a MOFid string, or None."""
    if not isinstance(s, str) or s.strip().lower() in BAD:
        return None
    return s.rsplit(";", 1)[-1].strip() if ";" in s else None


def build_groups(df: pd.DataFrame):
    """[KO] 3ì„±ë¶„ í‚¤ë¡œ ê·¸ë£¹ì„ ë§Œë“ ë‹¤ / [EN] build groups from the 3-part key."""
    ids = df[ID_COL].astype(str).tolist()
    metal_col = df[METAL_COL].map(
        lambda v: ",".join(sorted({x.strip() for x in str(v).replace(",", " ").split()
                                   if x.strip()})))
    P = {tag: [parse_mofid(v) for v in df[col]]
         for tag, col in (("v1", V1_COL), ("v2", V2_COL))}

    # [KO] ì¢…ë‹¨ ì‹ë³„ìž ê²€ì‚¬ (ì•„ëž˜ misaligned ê³„ì‚°ê³¼ ìˆœì„œ ë§žì¶¤)
    _mis = {"v1": set(), "v2": set()}
    if REFCODE_COL in df.columns:
        for tag, col in (("v1", V1_COL), ("v2", V2_COL)):
            for idx, (v, rc) in enumerate(zip(df[col], df[REFCODE_COL])):
                t = mofid_terminal_id(v)
                if t is not None and str(rc).strip() and t != str(rc).strip():
                    _mis[tag].add(idx)

    # [KO] topology í•©ì˜: v1 ìš°ì„ , ë¶ˆëª…ì´ë©´ v2. ì˜¤ì—¼ëœ ë²„ì „ì€ ì“°ì§€ ì•ŠëŠ”ë‹¤.
    # [EN] topology: v1 first, v2 fallback; misaligned versions are ignored.
    topo, n_conflict, n_fill = [], 0, 0
    for i, (a, b) in enumerate(zip(P["v1"], P["v2"])):
        t1 = a[2] if (a and i not in _mis["v1"]) else "*"
        t2 = b[2] if (b and i not in _mis["v2"]) else "*"
        if t1 != "*" and t2 != "*" and t1 != t2:
            n_conflict += 1
        if t1 == "*" and t2 != "*":
            n_fill += 1
        topo.append(t1 if t1 != "*" else t2)

    # [KO] ì¢…ë‹¨ ì‹ë³„ìž ê²€ì‚¬: MOFid ì˜ ';' ë’¤ ê°’ì´ ê·¸ í–‰ì˜ refcode ì™€ ë‹¬ë¼ì•¼ ì •ìƒì¸
    #      ì´ìœ ê°€ ì—†ë‹¤. ë‹¤ë¥´ë©´ ê·¸ MOFid ëŠ” ë‹¤ë¥¸ êµ¬ì¡°ì˜ ê°’ì´ë‹¤(í–‰ ë°€ë¦¼ ì˜¤ì—¼).
    # [EN] terminal-id check: the identifier after ';' must equal the row's own
    #      refcode. A mismatch means the MOFid belongs to another structure.
    misaligned = _mis

    st = dict(v1_key=0, v2_key=0, contaminated=0, pillar=0, misaligned=0,
              topo_conflict=n_conflict, topo_filled=n_fill, topo_unknown=0)

    dsu = DSU()
    for i in ids:
        dsu.find(("S", i))

    contaminated, pillar_rows = [], []
    for tag in ("v1", "v2"):
        for idx, (sid, r) in enumerate(zip(ids, P[tag])):
            if r is None:
                continue
            key_metals, linker, _ = r
            # [KO] ì¢…ë‹¨ ì‹ë³„ìžê°€ ë‹¤ë¥¸ êµ¬ì¡°ë¥¼ ê°€ë¦¬í‚¤ë©´ ê·¸ MOFid ëŠ” íê¸°
            # [EN] drop the MOFid if its terminal id points to another structure
            if idx in misaligned[tag]:
                st["misaligned"] += 1
                contaminated.append((sid, tag, "TERMINAL_ID_MISMATCH",
                                     str(df[REFCODE_COL].iloc[idx])))
                continue
            col_m = set(metal_col.iloc[idx].split(",")) - {""}

            # [KO] MOFid ê¸ˆì†ì´ ì»¬ëŸ¼ì˜ ë¶€ë¶„ì§‘í•©ì´ ì•„ë‹ˆë©´ ê·¸ MOFid ëŠ” ì˜¤ì—¼ -> íê¸°
            # [EN] if MOFid metals are not a subset of the column, the MOFid
            #      entry is contaminated -> drop this key
            if key_metals and col_m and not (key_metals <= col_m):
                st["contaminated"] += 1
                contaminated.append((sid, tag, ",".join(sorted(key_metals)),
                                     ",".join(sorted(col_m))))
                continue
            # [KO] ë¶€ë¶„ì§‘í•©ì´ë©´ MOFid ê°€ ê¸°ë‘¥ ê¸ˆì†ì„ ë¹ ëœ¨ë¦° ê²ƒ -> ì»¬ëŸ¼ ê°’ ì±„íƒ
            # [EN] a strict subset means MOFid dropped a pillar metal -> use column
            if key_metals and col_m and key_metals < col_m:
                st["pillar"] += 1
                pillar_rows.append((sid, tag, ",".join(sorted(key_metals)),
                                    ",".join(sorted(col_m))))
            # [KO] topology ë¶ˆëª…ì´ë©´ ê·¸ë£¹í™”ì— ì°¸ì—¬ì‹œí‚¤ì§€ ì•ŠëŠ”ë‹¤ (ë‹¨ë… ê·¸ë£¹)
            # [EN] unknown topology -> do not group this structure (singleton)
            if topo[idx] == "*":
                continue

            st[f"{tag}_key"] += 1
            dsu.union(("K", f"{tag}|{metal_col.iloc[idx]}|{linker}|{topo[idx]}"),
                      ("S", sid))

    st["topo_unknown"] = sum(1 for t in topo if t == "*")

    roots = {s: dsu.find(("S", s)) for s in ids}
    # [KO] ëŒ€í‘œ ë…¸ë“œë¥¼ ì •ë ¬í•´ ìž¬ë²ˆí˜¸ -> ê°™ì€ ìž…ë ¥ì´ë©´ í•­ìƒ ê°™ì€ group_id
    # [EN] renumber by sorted root -> deterministic group_id
    order = {r: i for i, r in enumerate(sorted(set(roots.values()), key=str))}
    g = pd.Series([order[roots[s]] for s in ids], index=df.index)

    cover = []
    for i, (a, b) in enumerate(zip(P["v1"], P["v2"])):
        if a is None and b is None:
            cover.append("no_MOFid")
        elif topo[i] == "*":
            cover.append("topology_unknown")
        else:
            cover.append("full_key")
    return g, pd.Series(cover, index=df.index), st, metal_col, contaminated, pillar_rows


# ============================================================
# [KO] ë¶„í•   /  [EN] splitting
# ============================================================

def make_strata(df, col, bins, log):
    """[KO] ê³„ì¸µí™” ê¸°ì¤€ ë°°ì—´. ì—°ì†í˜•ì´ë©´ ë¶„ìœ„ìˆ˜ë¡œ êµ¬ê°„í™”.
    [EN] Build the stratification array; continuous targets are quantile-binned."""
    s = df[col]
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > bins:
        q = pd.qcut(s.rank(method="first"), bins, labels=False, duplicates="drop")
        log(f"    '{col}' -> ì—°ì†í˜•, ë¶„ìœ„ìˆ˜ {int(pd.Series(q).nunique())}êµ¬ê°„ "
            f"/ continuous, {int(pd.Series(q).nunique())} quantile bins")
        return q.fillna(-1).astype(int).to_numpy()
    log(f"    '{col}' -> ë²”ì£¼í˜•, ê³ ìœ ê°’ {s.nunique()}ì¢… "
        f"/ categorical, {s.nunique()} classes")
    return s.astype("category").cat.codes.to_numpy()


def stratified_split(y, groups, seed):
    """[KO] ê·¸ë£¹ ì œì•½ + í´ëž˜ìŠ¤ ê· í˜• / [EN] group constraint + class balance."""
    from sklearn.model_selection import StratifiedGroupKFold
    skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    fold = np.empty(len(y), dtype=int)
    for k, (_, idx) in enumerate(skf.split(np.zeros(len(y)), y, groups)):
        fold[idx] = k
    return np.where(fold <= 7, "train", np.where(fold == 8, "val", "test"))


def random_group_split(groups, seed, n):
    """[KO] ê³„ì¸µí™” ì—†ì´ ê·¸ë£¹ ë‹¨ìœ„ ë¬´ìž‘ìœ„ ë°°ì • (target ê³¼ ë¬´ê´€).
    [EN] Group-level random assignment, independent of any target."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    rng.shuffle(ug)
    sizes = pd.Series(groups).value_counts()
    assign, cum = {}, 0
    for gid in ug:
        assign[gid] = "train" if cum < 0.8 * n else ("val" if cum < 0.9 * n else "test")
        cum += int(sizes[gid])
    return np.array([assign[x] for x in groups])


# ============================================================
# [KO] ë©”ì¸  /  [EN] main
# ============================================================

def run(args) -> int:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log = Log()

    if not Path(args.input).is_file():
        print(f"[STOP] ìž…ë ¥ íŒŒì¼ì´ ì—†ìŠµë‹ˆë‹¤ / input file not found: {args.input}")
        return 2
    df = pd.read_csv(args.input)

    # ---------- [KO] target ê²°ì • / [EN] resolve targets ----------
    labels: list[str] = []
    if args.target_file:
        # [KO] ë³„ë„ íŒŒì¼ì—ì„œ target ì„ ë¶™ì¸ë‹¤ (multi-target ì§€ì›)
        # [EN] join targets from a separate file (multi-target supported)
        if not Path(args.target_file).is_file():
            print(f"[STOP] target íŒŒì¼ ì—†ìŒ / target file not found: {args.target_file}")
            return 2
        tf = pd.read_csv(args.target_file)
        if ID_COL not in tf.columns:
            print(f"[STOP] target íŒŒì¼ì— '{ID_COL}' ì»¬ëŸ¼ì´ í•„ìš”í•©ë‹ˆë‹¤ "
                  f"/ target file needs a '{ID_COL}' column")
            print(f"       ì‹¤ì œ ì»¬ëŸ¼ / actual columns: {list(tf.columns)}")
            return 2
        labels = [c for c in tf.columns if c != ID_COL]
        dupcol = [c for c in labels if c in df.columns]
        if dupcol:
            df = df.drop(columns=dupcol)
        df = df.merge(tf, on=ID_COL, how="left")
    elif args.no_target:
        pass                                    # ëª…ì‹œì ìœ¼ë¡œ target ì—†ì´
    elif args.label_cols:
        labels = [c.strip() for c in args.label_cols.split(",") if c.strip()]
        missing_lab = [c for c in labels if c not in df.columns]
        if missing_lab:
            print("[STOP] --label-cols ì— ì§€ì •í•œ ì»¬ëŸ¼ì´ CSV ì— ì—†ìŠµë‹ˆë‹¤"
                  " / columns given in --label-cols are not in the CSV:")
            for c in missing_lab:
                print(f"         {c}")
            print("\n       CSV ì˜ ì‹¤ì œ ì»¬ëŸ¼ / actual columns:")
            for c in df.columns:
                print(f"         {c}")
            return 2
    else:
        # [KO] --label-cols ë„ --target-file ë„ ì—†ìœ¼ë©´ target ì—†ì´ split ë§Œ ë§Œë“ ë‹¤.
        #      ë¶„ë¥˜ê°€ ì•„ë‹Œ ê³¼ì œ(íšŒê·€Â·ë¶„í¬ ì˜ˆì¸¡ ë“±)ì—ì„œëŠ” ì´ê²Œ ì •ìƒ ì‚¬ìš©ë²•ì´ë‹¤.
        # [EN] With neither --label-cols nor --target-file, build the split only.
        #      This is the normal path for non-classification tasks.
        auto = [c for c in ("KH_Label_quaternary", "KH_Label_binary")
                if c in df.columns]
        if auto:
            labels = auto

    strat = args.stratify_on or (labels[0] if labels else None)

    # Require refcode so terminal-ID contamination validation cannot be skipped silently.
    need = [ID_COL, REFCODE_COL, V1_COL, V2_COL, METAL_COL] + labels
    if strat:
        need.append(strat)
    miss = [c for c in dict.fromkeys(need) if c not in df.columns]
    if miss:
        log("[STOP] ì•„ëž˜ ì»¬ëŸ¼ì´ ì—†ìŠµë‹ˆë‹¤ / the following columns are missing:")
        for c in miss:
            log(f"         {c}")
        log("\n       ì‹¤ì œ ì»¬ëŸ¼ / actual columns:")
        for c in df.columns:
            log(f"         {c}")
        log.save(outdir / "split_log.txt")
        return 2

    log("=" * 74)
    log(f"ìž…ë ¥ / input   : {args.input}   ({len(df)} rows x {df.shape[1]} cols)")
    if args.target_file:
        log(f"target íŒŒì¼    : {args.target_file}")
    if labels:
        log(f"target         : {', '.join(labels)}")
    else:
        log("target         : ì—†ìŒ / none")
        log("                 split ë§Œ ë§Œë“ ë‹¤. ë‚˜ì¤‘ì— split.json ìœ¼ë¡œ ë¶™ì´ë©´ ëœë‹¤.")
        log("                 Building the split only; join targets later via split.json.")
    log("=" * 74)

    # [KO] OMS ê°€ ì„žì—¬ ìžˆìœ¼ë©´ ìžë™ ì œê±° / [EN] auto-remove OMS-bearing structures
    if OMS_COL in df.columns:
        flag = df[OMS_COL].astype(str).str.strip().str.lower().map(OMS_MAP)
        if flag.isna().any():
            log(f"[STOP] '{OMS_COL}' ì— í•´ì„ ë¶ˆê°€ ê°’ / unparsable values: "
                f"{df.loc[flag.isna(), OMS_COL].unique()[:5]}")
            log.save(outdir / "split_log.txt")
            return 2
        n_oms = int(flag.sum())
        if n_oms:
            df = df[~flag].copy().reset_index(drop=True)
            log(f"  OMS ë³´ìœ  {n_oms}ê±´ ìžë™ ì œê±° / removed {n_oms} OMS structures "
                f"-> {len(df)}")
        else:
            log("  OMS ë³´ìœ  êµ¬ì¡° ì—†ìŒ / no OMS structures (checked)")

    ndup = int(df[ID_COL].duplicated().sum())
    log(f"  {ID_COL} ì¤‘ë³µ / duplicated : {ndup}" + ("   <-- check" if ndup else ""))
    for c in labels:
        n_na = int(df[c].isna().sum())
        if n_na:
            log(f"  [warn] '{c}' ê²°ì¸¡ {n_na}ê±´ / {n_na} missing "
                "- í•´ë‹¹ target íŒŒì¼ì—ì„œ ì œì™¸ / excluded from that target file")

    # ---------- STEP 1 ----------
    log("\n" + "=" * 74)
    log("STEP 1  MOFid ë¡œ ì¤‘ë³µ ë¬¶ê¸° / group duplicates by MOFid")
    log("        í‚¤ / key = metal | linker | topology")
    log("=" * 74)
    g, cover, st, metal_col, contaminated, pillar_rows = build_groups(df)
    df["group_id"], df["key_coverage"], df["_metal"] = g, cover, metal_col

    log("  [metal]  Metal Types ì»¬ëŸ¼ í™•ì • + MOFid êµì°¨ ê²€ì¦"
        " / column is authoritative, MOFid cross-checked")
    log(f"    MOFid ê°€ ê¸ˆì† ëˆ„ë½ -> ì»¬ëŸ¼ ì±„íƒ / MOFid missed a metal   {st['pillar']:5d}")
    log(f"    ì¢…ë‹¨ ì‹ë³„ìž ë¶ˆì¼ì¹˜ -> í‚¤ íê¸°   / terminal id mismatch   {st['misaligned']:5d}")
    log(f"    ê¸ˆì† ë¶ˆì¼ì¹˜ -> í‚¤ íê¸°          / metal mismatch         {st['contaminated']:5d}")
    log("  [topology]  v1 ìš°ì„ , ë¶ˆëª…ì´ë©´ v2 / v1 first, v2 fallback")
    log(f"    v2 ë¡œ ë³´ì™„                     / filled from v2         {st['topo_filled']:5d}")
    log(f"    v1Â·v2 ë¶ˆì¼ì¹˜ -> v1 ì±„íƒ        / conflict, v1 taken     {st['topo_conflict']:5d}")
    log(f"    ë‘˜ ë‹¤ ë¶ˆëª… -> ê·¸ë£¹í™” ì œì™¸      / both unknown, excluded {st['topo_unknown']:5d}")
    log(f"  [keys]  v1 {st['v1_key']}   v2 {st['v2_key']}")

    cc = cover.value_counts()
    log("\n  [êµ¬ì¡°ë³„ ìƒíƒœ / structure status]")
    for k, ko in (("full_key", "ì™„ì „ í‚¤"), ("topology_unknown", "topology ë¶ˆëª…"),
                  ("no_MOFid", "MOFid ì—†ìŒ")):
        n = int(cc.get(k, 0))
        note = "ê·¸ë£¹í™” ì°¸ì—¬ / grouped" if k == "full_key" else "ë‹¨ë… ê·¸ë£¹ / singleton"
        log(f"    {ko:14s} {k:18s} {n:5d}  ({100*n/len(df):5.1f}%)   {note}")

    n_group = int(df["group_id"].nunique())
    n_dup = len(df) - n_group
    sz = df["group_id"].value_counts()
    log("\n  [ê²°ê³¼ / result]")
    log(f"    êµ¬ì¡° / structures  {len(df)}")
    log(f"    ê·¸ë£¹ / groups      {n_group}")
    log(f"    ì¤‘ë³µ / duplicates  {n_dup}  ({100*n_dup/len(df):.1f}%)")
    log(f"    ìµœëŒ€ ê·¸ë£¹ / largest {int(sz.max())}   í‰ê·  / mean {sz.mean():.2f}")
    log("\n  [ê·¸ë£¹ í¬ê¸° ë¶„í¬ / group size distribution]")
    log(f"    {'size':>6s} {'groups':>8s} {'structures':>11s} {'share':>8s}")
    for lo, hi, lab in [(1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"),
                        (6, 10, "6-10"), (11, 10**9, "11+")]:
        gs = sz[(sz >= lo) & (sz <= hi)]
        log(f"    {lab:>6s} {len(gs):8d} {int(gs.sum()):11d} {100*gs.sum()/len(df):7.1f}%")

    viol = int((df.groupby("group_id")["_metal"].nunique() > 1).sum())
    log(f"\n  [check] ê·¸ë£¹ ë‚´ ê¸ˆì† ë¶ˆì¼ì¹˜ / groups with mixed metals : {viol}"
        + ("  <-- check" if viol else "  (OK)"))

    # ---------- STEP 2 ----------
    log("\n" + "=" * 74)
    log(f"STEP 2  8 : 1 : 1 ë¶„í•  / split   seed={args.seed}")
    log("=" * 74)
    if strat is None or args.stratify == "none" or args.random_split:
        if strat is None:
            log("  target ì´ ì—†ì–´ ê³„ì¸µí™” ì—†ì´ ê·¸ë£¹ ë‹¨ìœ„ ë¬´ìž‘ìœ„ ë°°ì •")
            log("  no target -> group-level random assignment (target-independent)")
        elif args.random_split:
            log("  --random-split : target ì„ ë³´ì§€ ì•Šê³  ê·¸ë£¹ ë‹¨ìœ„ ë¬´ìž‘ìœ„ ë°°ì •")
            log("  --random-split : groups assigned at random, target ignored")
        else:
            log("  ê³„ì¸µí™” ì—†ì´ ê·¸ë£¹ ë‹¨ìœ„ ë¬´ìž‘ìœ„ ë°°ì • / stratification disabled")
        df["split"] = random_group_split(df["group_id"].to_numpy(), args.seed, len(df))
    else:
        log("  ê³„ì¸µí™” ì‚¬ìš© - target ë¶„í¬ë¥¼ ê³ ë¥´ê²Œ ë§žì¶˜ë‹¤")
        log("  stratified - target distribution is balanced across splits")
        y = make_strata(df, strat, args.stratify_bins, log)
        df["split"] = stratified_split(y, df["group_id"].to_numpy(), args.seed)

    assert int((df.groupby("group_id")["split"].nunique() > 1).sum()) == 0, \
        "group spans multiple splits"

    log(f"\n    {'split':8s} {'structures':>11s} {'share':>8s} {'groups':>8s}")
    for s in ("train", "val", "test"):
        sub = df[df["split"] == s]
        log(f"    {s:8s} {len(sub):11d} {100*len(sub)/len(df):7.2f}% "
            f"{sub['group_id'].nunique():8d}")
    log(f"    {'total':8s} {len(df):11d} {'100.00%':>8s} {n_group:8d}")
    log("\n    [check] ì—¬ëŸ¬ split ì— ê±¸ì¹œ ê·¸ë£¹ / groups spanning splits : 0  (OK)")

    for col in labels:
        s = df[col].dropna()
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > args.stratify_bins:
            log(f"\n  [target ë¶„í¬ / distribution - {col}]  ì—°ì†í˜• / continuous")
            log(f"    {'split':8s} {'n':>6s} {'mean':>11s} {'std':>11s} "
                f"{'min':>11s} {'max':>11s}")
            for sp in ("train", "val", "test"):
                v = df.loc[df["split"] == sp, col].dropna()
                if not len(v):
                    continue
                log(f"    {sp:8s} {len(v):6d} {v.mean():11.4g} {v.std():11.4g} "
                    f"{v.min():11.4g} {v.max():11.4g}")
            log(f"    {'total':8s} {len(s):6d} {s.mean():11.4g} {s.std():11.4g} "
                f"{s.min():11.4g} {s.max():11.4g}")
            continue

        log(f"\n  [í´ëž˜ìŠ¤ ë¶„í¬ / class distribution - {col}]")
        t = pd.crosstab(df[col], df["split"])
        for c in ("train", "val", "test"):
            if c not in t:
                t[c] = 0
        t = t[["train", "val", "test"]]
        share = pd.crosstab(df[col], df["split"], normalize="columns") * 100
        overall = df[col].value_counts(normalize=True).sort_index() * 100
        log(f"    {'label':>8s} {'train':>7s} {'val':>6s} {'test':>6s} {'total':>7s}"
            f" | {'train%':>7s} {'val%':>6s} {'test%':>6s} {'all%':>6s}")
        for k in t.index:
            log(f"    {str(k):>8s} {t.loc[k,'train']:7d} {t.loc[k,'val']:6d} "
                f"{t.loc[k,'test']:6d} {int(t.loc[k].sum()):7d} |"
                f" {share.loc[k,'train']:6.2f}% {share.loc[k,'val']:5.2f}%"
                f" {share.loc[k,'test']:5.2f}% {overall[k]:5.2f}%")
        dev = max(abs(share[c] - overall).max() for c in share.columns)
        log(f"    ì „ì²´ ëŒ€ë¹„ ìµœëŒ€ íŽ¸ì°¨ / max deviation from overall  {dev:.2f}%p")
        small = t[t["test"] < 10]
        if len(small):
            log(f"    [warn] test í‘œë³¸ 10ê°œ ë¯¸ë§Œ / fewer than 10 test samples: "
                f"{list(small.index)}")

    log("\n  [êµ¬ì¡° ìƒíƒœë³„ split / split by structure status]")
    t = pd.crosstab(df["key_coverage"], df["split"])
    for c in ("train", "val", "test"):
        if c not in t:
            t[c] = 0
    log(f"    {'status':18s} {'train':>7s} {'val':>6s} {'test':>6s} {'total':>7s}")
    for k in t.index:
        r = t.loc[k]
        log(f"    {k:18s} {r['train']:7d} {r['val']:6d} {r['test']:6d} {int(r.sum()):7d}")

    # ---------- STEP 3 ----------
    log("\n" + "=" * 74)
    log("STEP 3  ì €ìž¥ / save")
    log("=" * 74)
    js = {s: sorted(df.loc[df["split"] == s, ID_COL].astype(str))
          for s in ("train", "val", "test")}
    meta = dict(seed=args.seed, ratio="8:1:1", label_cols=labels,
                stratify=("none" if (strat is None or args.random_split)
                          else args.stratify),
                stratify_on=(None if (strat is None or args.random_split
                                      or args.stratify == "none") else strat),
                group_key="MOFid metal|linker|topology (v1+v2 union-find)",
                topology_unknown="singleton",
                n_structures=len(df), n_groups=n_group, n_duplicates=n_dup,
                counts={k: len(v) for k, v in js.items()})
    (outdir / "split.json").write_text(
        json.dumps({**meta, **js}, ensure_ascii=False, indent=1), encoding="utf-8")
    log("  split.json                    " + "  ".join(f"{k}={len(v)}" for k, v in js.items()))

    df[[ID_COL, "group_id", "split", "key_coverage", "_metal"] + labels] \
        .rename(columns={"_metal": "metal_types"}) \
        .to_csv(outdir / "split_assignment.csv", index=False)
    log(f"  split_assignment.csv          ({len(df)} rows)")

    # [KO] MEFNet dataset.py í˜¸í™˜ labels íŒŒì¼ (split ë¬´ê´€, ì „ì²´ êµ¬ì¡°)
    #      ì»¬ëŸ¼ëª…ì„ mof_id / label ë¡œ ë§žì¶˜ë‹¤.
    # [EN] labels file for MEFNet dataset.py (all structures, split-independent);
    #      columns renamed to mof_id / label.
    for col in labels:
        tag = re.sub(r"[^0-9A-Za-z]+", "_", col).strip("_")
        lab = df[[ID_COL, col]].dropna().copy()
        lab.columns = ["mof_id", "label"]
        if pd.api.types.is_numeric_dtype(lab["label"]) and \
           len(lab) and (lab["label"] % 1 == 0).all():
            lab["label"] = lab["label"].astype(int)
        if args.logkh_col and args.logkh_col in df.columns:
            extra = df[[ID_COL, args.logkh_col]].rename(
                columns={ID_COL: "mof_id", args.logkh_col: "log_kh"})
            lab = lab.merge(extra, on="mof_id", how="left")
        lab.to_csv(outdir / f"labels_{tag}.csv", index=False)
        log(f"  labels_{tag}.csv   ({len(lab)} rows)  "
            f"-> MEFNet dataset.py ìš© / for MEFNet")

    for col in labels:
        tag = re.sub(r"[^0-9A-Za-z]+", "_", col).strip("_")
        for sp in ("train", "val", "test"):
            sub = df.loc[df["split"] == sp, [ID_COL, col]].dropna()
            sub.columns = [ID_COL, "target"]
            if pd.api.types.is_numeric_dtype(sub["target"]) and \
               len(sub) and (sub["target"] % 1 == 0).all():
                sub["target"] = sub["target"].astype(int)
            sub.to_csv(outdir / f"id_prop_{tag}_{sp}.csv",
                       index=False, header=not args.no_header)
        log(f"  id_prop_{tag}_{{train,val,test}}.csv")
    if not labels:
        log("  (target ì—†ìŒ -> id_prop íŒŒì¼ ì—†ìŒ / no target -> no id_prop files)")
        log("   split.json ë˜ëŠ” split_assignment.csv ë¡œ ë‚˜ì¤‘ì— ë¶™ì´ë©´ ëœë‹¤")
        log("   join your targets later using split.json or split_assignment.csv")

    grp = df.groupby("group_id").agg(
        n=(ID_COL, "size"), split=("split", "first"), metal=("_metal", "first"),
        coreids=(ID_COL, lambda s: ";".join(map(str, s))))
    grp[grp["n"] > 1].sort_values("n", ascending=False).to_csv(
        outdir / "duplicate_groups.csv")
    log(f"  duplicate_groups.csv          ({int((sz>1).sum())} groups)")

    if contaminated:
        pd.DataFrame(contaminated, columns=[ID_COL, "mofid_version",
                                            "mofid_metals", "metal_types"]) \
            .to_csv(outdir / "check_contaminated_mofid.csv", index=False)
        log(f"  check_contaminated_mofid.csv  ({len(contaminated)} rows)")
    if pillar_rows:
        pd.DataFrame(pillar_rows, columns=[ID_COL, "mofid_version",
                                           "mofid_metals", "metal_types"]) \
            .to_csv(outdir / "check_missing_metal.csv", index=False)
        log(f"  check_missing_metal.csv       ({len(pillar_rows)} rows)")

    log.save(outdir / "split_log.txt")
    print(f"\n  ì „ì²´ ê¸°ë¡ / full log: {outdir/'split_log.txt'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="CoRE MOF group split 8:1:1  (MOFid-based duplicate grouping)",
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", required=True,
                   help="ìž…ë ¥ CSV / input CSV")
    p.add_argument("--outdir", default="splits",
                   help="ê²°ê³¼ í´ë” / output folder (default: splits)")
    p.add_argument("--label-cols", default=None,
                   help="target ì»¬ëŸ¼ëª…, ì‰¼í‘œ êµ¬ë¶„ / target columns, comma-separated\n"
                        "  ìƒëžµí•˜ë©´ target ì—†ì´ split ë§Œ ë§Œë“ ë‹¤\n"
                        "  omit to build the split without any target")
    p.add_argument("--target-file", default=None,
                   help="target ì´ ë‹´ê¸´ ë³„ë„ CSV / separate CSV holding targets\n"
                        "  coreid + target ì»¬ëŸ¼ë“¤ / coreid + one or more target columns\n"
                        "  multi-target ì€ ì´ ë°©ì‹ì„ ì“°ë©´ ëœë‹¤ / use this for multi-target")
    p.add_argument("--no-target", action="store_true",
                   help="target ì—†ì´ split ë§Œ ë§Œë“ ë‹¤ / build the split without any target")
    p.add_argument("--random-split", action="store_true",
                   help="target ì„ ë³´ì§€ ì•Šê³  ê·¸ë£¹ ë‹¨ìœ„ ë¬´ìž‘ìœ„ ë°°ì •\n"
                        "/ ignore the target, assign groups at random\n"
                        "  [KO] target ì´ ë¶„í¬Â·ê°€ë³€ê¸¸ì´ ë“± ìŠ¤ì¹¼ë¼ê°€ ì•„ë‹ ë•Œ\n"
                        "  [EN] use when the target is not a scalar\n"
                        "       (e.g. a distribution of variable length)")
    p.add_argument("--seed", type=int, default=42,
                   help="ë‚œìˆ˜ seed / random seed (default: 42)")
    p.add_argument("--logkh-col", default=None,
                   help="log KH íšŒê·€ìš© ì»¬ëŸ¼ / column for log-KH regression\n"
                        "  labels_*.csv ì— log_kh ì—´ë¡œ í•¨ê»˜ ì €ìž¥ëœë‹¤")
    p.add_argument("--no-header", action="store_true",
                   help="id_prop ì„ í—¤ë” ì—†ì´ ì €ìž¥ / write id_prop without a header row")
    # [KO] ê³ ê¸‰ ì˜µì…˜ - ë³´í†µ ê±´ë“œë¦´ í•„ìš” ì—†ìŒ / [EN] advanced - rarely needed
    p.add_argument("--stratify", default="auto", choices=["auto", "none"],
                   help=argparse.SUPPRESS)
    p.add_argument("--stratify-on", default=None, help=argparse.SUPPRESS)
    p.add_argument("--stratify-bins", type=int, default=10, help=argparse.SUPPRESS)
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())

