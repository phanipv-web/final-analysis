#!/usr/bin/env python3
"""
InterProScan TSV summarizer (supports separate dbCAN and MEROPS TSVs)

Inputs:
  - main InterProScan TSV (required)
  - dbCAN TSV (optional)  -> for CAZyme summaries
  - MEROPS TSV (optional) -> for protease family summaries
  - proteins FASTA (optional) -> for total proteins + proteins_without_hit

Outputs (TSV):
  global_stats.tsv
  analysis_contribution.tsv
  interpro_abundance.tsv
  signature_abundance.tsv
  go_term_counts.tsv
  go_protein_counts.tsv
  cazymes_hit_counts.tsv
  cazymes_protein_counts.tsv
  protease_hit_counts.tsv
  protease_protein_counts.tsv
  protease_families.tsv           (MEROPS-only if merops TSV provided)
  protease_class_counts.tsv       (classify serine/metallo/cysteine/aspartic/threonine; keyword-based)
  ec_numbers.tsv                  (best-effort, often sparse in TSV)
  domains_per_protein.tsv
  unannotated_proteins.ids        (if --fasta provided)

Notes:
- Treats '-', 'NA', 'null' etc. as missing.
- Fixes the "interpro_acc = '-'" bug you observed in your current outputs.
- dbCAN/MEROPS TSVs can be InterProScan-like TSV (13–15 cols) OR
  simple 3-col (protein_id, family, desc). If your dbCAN/MEROPS format differs,
  tell me the first line (head -n 1 | cat -A) and I’ll adjust.

Usage examples:

1) Main only:
  python scripts/interproscan_summary.py -i data/interproscan.tsv -o results --fasta data/proteins.fasta

2) Main + dbCAN + MEROPS:
  python scripts/interproscan_summary.py \
    -i data/interproscan.tsv \
    --dbcan data/interproscan_dbcan.tsv \
    --merops data/interproscan_merops.tsv \
    --fasta data/proteins.fasta \
    -o results
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Column handling
# -----------------------------
STD_COLS_15 = [
    "protein_id", "md5", "seq_len", "analysis",
    "signature_acc", "signature_desc",
    "start", "end", "score", "status", "date",
    "interpro_acc", "interpro_desc",
    "go_terms", "pathways"
]
STD_COLS_14 = STD_COLS_15[:-1]  # no pathways
STD_COLS_13 = STD_COLS_15[:-2]  # no GO, no pathways

MISSING = {"", "-", "NA", "NaN", "nan", "null", "None", "none"}


def norm_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s in MISSING else s


def is_present(x) -> bool:
    return norm_str(x) != ""


def split_terms(s: str) -> List[str]:
    s = norm_str(s)
    if not s:
        return []
    parts = re.split(r"[|,;]+", s)
    return [p.strip() for p in parts if p.strip()]


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False)


def read_tsv_flexible(path: Path, label: str) -> pd.DataFrame:
    """
    Reads:
    - InterProScan-style TSV with 13–15 columns (no header)
    OR
    - a simple TSV with columns like: protein_id, family, desc (>=2 cols)
    """
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, na_filter=False)
    ncol = df.shape[1]

    # InterProScan TSV format
    if ncol >= 13:
        if ncol >= 15:
            df = df.iloc[:, :15]
            df.columns = STD_COLS_15
        elif ncol == 14:
            df.columns = STD_COLS_14
            df["pathways"] = ""
        elif ncol == 13:
            df.columns = STD_COLS_13
            df["go_terms"] = ""
            df["pathways"] = ""

        # Normalize string columns
        for c in ["protein_id","analysis","signature_acc","signature_desc","interpro_acc","interpro_desc","go_terms","pathways"]:
            if c in df.columns:
                df[c] = df[c].map(norm_str)

        # numeric-ish fields
        df["seq_len"] = pd.to_numeric(df.get("seq_len", ""), errors="coerce")
        df["start"] = pd.to_numeric(df.get("start", ""), errors="coerce")
        df["end"] = pd.to_numeric(df.get("end", ""), errors="coerce")

        df["source"] = label
        return df

    # Simple fallback format (protein_id, family, desc...)
    if ncol < 2:
        raise ValueError(f"{path} has {ncol} columns; not enough to parse.")

    # Map to "InterProScan-like" minimal columns
    out = pd.DataFrame()
    out["protein_id"] = df.iloc[:, 0].map(norm_str)
    out["analysis"] = label  # treat as analysis label
    out["signature_acc"] = df.iloc[:, 1].map(norm_str)
    out["signature_desc"] = df.iloc[:, 2].map(norm_str) if ncol >= 3 else ""
    out["md5"] = ""
    out["seq_len"] = np.nan
    out["start"] = np.nan
    out["end"] = np.nan
    out["score"] = ""
    out["status"] = ""
    out["date"] = ""
    out["interpro_acc"] = ""
    out["interpro_desc"] = ""
    out["go_terms"] = ""
    out["pathways"] = ""
    out["source"] = label
    return out


def extract_ids_from_fasta(fasta: Path) -> pd.Index:
    ids = []
    with fasta.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                ids.append(line[1:].strip().split()[0])
    return pd.Index(pd.unique(ids))


# -----------------------------
# Summaries
# -----------------------------
def global_stats(df_all: pd.DataFrame, all_proteins: Optional[pd.Index]) -> pd.DataFrame:
    """
    Global stats are computed from all rows combined (main + dbCAN + merops),
    BUT key annotation fields (GO/InterPro) usually only make sense from main TSV.
    We'll compute:
      - proteins_with_any_hit from combined
      - proteins_with_interpro/go/pathway from combined but these are typically main-driven
    """
    proteins_with_any_hit = pd.Index(df_all["protein_id"].unique())

    total_proteins = len(all_proteins) if all_proteins is not None else len(proteins_with_any_hit)
    annotated_proteins = len(proteins_with_any_hit)
    proteins_without_hit = (total_proteins - annotated_proteins) if all_proteins is not None else np.nan

    proteins_with_interpro = df_all.loc[df_all["interpro_acc"].map(is_present), "protein_id"].nunique()
    proteins_with_go = df_all.loc[df_all["go_terms"].map(is_present), "protein_id"].nunique()
    proteins_with_pathway = df_all.loc[df_all["pathways"].map(is_present), "protein_id"].nunique()

    rows_total = len(df_all)
    hit_rows_with_interpro = int(df_all["interpro_acc"].map(is_present).sum())
    hit_rows_with_go = int(df_all["go_terms"].map(is_present).sum())
    hit_rows_with_pathway = int(df_all["pathways"].map(is_present).sum())

    dpp = df_all.groupby("protein_id").size()

    out = pd.DataFrame([{
        "total_proteins": total_proteins,
        "proteins_with_any_hit": annotated_proteins,
        "proteins_without_hit": proteins_without_hit,
        "total_hit_rows": rows_total,
        "proteins_with_interpro": proteins_with_interpro,
        "proteins_with_go": proteins_with_go,
        "proteins_with_pathway": proteins_with_pathway,
        "hit_rows_with_interpro": hit_rows_with_interpro,
        "hit_rows_with_go": hit_rows_with_go,
        "hit_rows_with_pathway": hit_rows_with_pathway,
        "mean_hits_per_protein": float(dpp.mean()) if len(dpp) else 0.0,
        "median_hits_per_protein": float(dpp.median()) if len(dpp) else 0.0,
        "max_hits_per_protein": int(dpp.max()) if len(dpp) else 0,
    }])
    return out


def analysis_contribution(df_all: pd.DataFrame) -> pd.DataFrame:
    g = df_all.groupby("analysis")
    out = pd.DataFrame({
        "hit_rows": g.size(),
        "unique_proteins": g["protein_id"].nunique(),
        "unique_signatures": g["signature_acc"].apply(lambda x: len({norm_str(v) for v in x if is_present(v)})),
        "unique_interpro_hits": g["interpro_acc"].apply(lambda x: int(pd.Series(x).map(is_present).sum())),
    }).reset_index()
    return out.sort_values("hit_rows", ascending=False)


def abundance_tables(df_main: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # InterPro abundance (main only)
    ip = df_main[df_main["interpro_acc"].map(is_present)].copy()
    interpro_ab = (
        ip.groupby(["interpro_acc", "interpro_desc"])
          .agg(hit_rows=("interpro_acc", "size"),
               unique_proteins=("protein_id", "nunique"))
          .reset_index()
          .sort_values(["unique_proteins", "hit_rows"], ascending=False)
    )

    # Signature abundance (main only)
    sig = df_main[df_main["signature_acc"].map(is_present)].copy()
    sig_ab = (
        sig.groupby(["analysis", "signature_acc", "signature_desc"])
          .agg(hit_rows=("signature_acc", "size"),
               unique_proteins=("protein_id", "nunique"))
          .reset_index()
          .sort_values(["unique_proteins", "hit_rows"], ascending=False)
    )
    return interpro_ab, sig_ab


def go_summaries(df_main: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, r in df_main.iterrows():
        for go in split_terms(r.get("go_terms", "")):
            # keep source tags like GO:xxx(InterPro) as-is; still counts fine
            if go.startswith("GO:"):
                rows.append((go, r["protein_id"]))

    if not rows:
        return (pd.DataFrame(columns=["go_term", "hit_rows", "unique_proteins"]),
                pd.DataFrame(columns=["go_term", "unique_proteins"]))

    go_df = pd.DataFrame(rows, columns=["go_term", "protein_id"])
    go_hit = (go_df.groupby("go_term")
                    .agg(hit_rows=("go_term", "size"),
                         unique_proteins=("protein_id", "nunique"))
                    .reset_index()
                    .sort_values(["unique_proteins", "hit_rows"], ascending=False))
    go_prot = (go_df.groupby("go_term")["protein_id"]
                     .nunique()
                     .reset_index(name="unique_proteins")
                     .sort_values("unique_proteins", ascending=False))
    return go_hit, go_prot


# CAZy family regex (dbCAN usually provides GHxx/GTxx/CE/AA/PL/CBM)
CAZY_FAMILY_RE = re.compile(r"^(GH|GT|PL|CE|AA|CBM)\d+([A-Za-z0-9_.-]+)?$")


def cazymes(df_dbcan: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df_dbcan is None or df_dbcan.empty:
        cols = ["cazy_family", "analysis", "hit_rows", "unique_proteins", "signature_desc"]
        return (pd.DataFrame(columns=cols),
                pd.DataFrame(columns=["cazy_family", "unique_proteins"]))

    # Keep only things that look like CAZy families, but also allow everything if you prefer.
    d = df_dbcan.copy()
    fam = d["signature_acc"].map(norm_str)
    d = d[fam.map(lambda x: bool(CAZY_FAMILY_RE.match(x)) if x else False)].copy()

    if d.empty:
        cols = ["cazy_family", "analysis", "hit_rows", "unique_proteins", "signature_desc"]
        return (pd.DataFrame(columns=cols),
                pd.DataFrame(columns=["cazy_family", "unique_proteins"]))

    d["cazy_family"] = d["signature_acc"]
    hit_counts = (d.groupby(["cazy_family", "analysis", "signature_desc"])
                    .agg(hit_rows=("cazy_family", "size"),
                         unique_proteins=("protein_id", "nunique"))
                    .reset_index()
                    .sort_values(["unique_proteins", "hit_rows"], ascending=False))
    prot_counts = (d.groupby("cazy_family")["protein_id"]
                    .nunique()
                    .reset_index(name="unique_proteins")
                    .sort_values("unique_proteins", ascending=False))
    return hit_counts, prot_counts


# Protease class keywording (works even without MEROPS family codes)
PROTEASE_CLASS_PATTERNS = [
    ("serine_protease", re.compile(r"\bserine\b|\bsubtilisin\b|\btrypsin\b", re.I)),
    ("metalloprotease", re.compile(r"\bmetallo\b|\bzinc\b|\bmatrix metalloproteinase\b", re.I)),
    ("cysteine_protease", re.compile(r"\bcysteine\b|\bpapain\b|\bcathepsin\b", re.I)),
    ("aspartic_protease", re.compile(r"\baspartic\b|\bpepsin\b", re.I)),
    ("threonine_protease", re.compile(r"\bthreonine\b|\bproteasome\b", re.I)),
]


def proteases_keyword_table(df_all: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Identify protease-like rows (keyword-based) OR MEROPS analysis rows if present in combined
    blob = (df_all["signature_desc"].fillna("") + " " + df_all["interpro_desc"].fillna("")).astype(str)
    is_prot = blob.str.contains(r"\b(peptidase|protease)\b", case=False, regex=True)

    # also include InterPro peptidase domains even if desc lacks keywords (your data has them already)
    is_peptidase_ipr = df_all["interpro_desc"].astype(str).str.contains(r"\bpeptidase\b", case=False, regex=True)

    p = df_all[is_prot | is_peptidase_ipr].copy()
    if p.empty:
        base_cols = ["protease_key", "analysis", "signature_desc", "interpro_acc", "interpro_desc", "hit_rows", "unique_proteins"]
        return (
            pd.DataFrame(columns=base_cols),
            pd.DataFrame(columns=["protease_key", "unique_proteins"]),
            pd.DataFrame(columns=["protease_class", "hit_rows", "unique_proteins"])
        )

    # Key: prefer InterPro ID if present; else signature_acc
    p["protease_key"] = np.where(
        p["interpro_acc"].map(is_present),
        p["interpro_acc"],
        p["signature_acc"]
    )

    hit_counts = (p.groupby(["protease_key", "analysis", "signature_desc", "interpro_acc", "interpro_desc"])
                    .agg(hit_rows=("protease_key", "size"),
                         unique_proteins=("protein_id", "nunique"))
                    .reset_index()
                    .sort_values(["unique_proteins", "hit_rows"], ascending=False))

    prot_counts = (p.groupby("protease_key")["protein_id"]
                    .nunique()
                    .reset_index(name="unique_proteins")
                    .sort_values("unique_proteins", ascending=False))

    # Protease class table
    def classify(row) -> str:
        text = f"{row.get('signature_desc','')} {row.get('interpro_desc','')}"
        for name, pat in PROTEASE_CLASS_PATTERNS:
            if pat.search(text):
                return name
        return "unclassified_protease"

    p["protease_class"] = p.apply(classify, axis=1)
    class_tbl = (p.groupby("protease_class")
                  .agg(hit_rows=("protease_class", "size"),
                       unique_proteins=("protein_id", "nunique"))
                  .reset_index()
                  .sort_values(["unique_proteins", "hit_rows"], ascending=False))
    return hit_counts, prot_counts, class_tbl


def merops_families(df_merops: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df_merops is None or df_merops.empty:
        return pd.DataFrame(columns=["protease_family", "analysis", "hit_rows", "unique_proteins", "signature_desc"])

    m = df_merops.copy()
    m = m[m["signature_acc"].map(is_present)].copy()
    if m.empty:
        return pd.DataFrame(columns=["protease_family", "analysis", "hit_rows", "unique_proteins", "signature_desc"])

    # signature_acc should be the MEROPS family/accession in InterProScan MEROPS output
    m["protease_family"] = m["signature_acc"]
    tbl = (m.groupby(["protease_family", "analysis", "signature_desc"])
             .agg(hit_rows=("protease_family", "size"),
                  unique_proteins=("protein_id", "nunique"))
             .reset_index()
             .sort_values(["unique_proteins", "hit_rows"], ascending=False))
    return tbl


EC_RE = re.compile(r"\bEC[:\s]*([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)\b")


def ec_numbers(df_main: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_main.iterrows():
        blob = " ".join([
            str(r.get("signature_desc", "")),
            str(r.get("interpro_desc", "")),
            str(r.get("pathways", "")),
        ])
        ecs = EC_RE.findall(blob)
        if not ecs:
            continue
        for ec in set(ecs):
            rows.append((ec, r["protein_id"], r.get("analysis", ""), r.get("interpro_acc", ""), r.get("signature_acc", "")))

    if not rows:
        return pd.DataFrame(columns=["ec_number", "unique_proteins", "hit_rows", "analyses", "example_interpro", "example_signature"])

    e = pd.DataFrame(rows, columns=["ec_number", "protein_id", "analysis", "interpro_acc", "signature_acc"])
    out = (e.groupby("ec_number")
             .agg(unique_proteins=("protein_id", "nunique"),
                  hit_rows=("ec_number", "size"),
                  analyses=("analysis", lambda x: ",".join(sorted(set([a for a in x if a])))),
                  example_interpro=("interpro_acc", lambda x: next((v for v in x if is_present(v)), "")),
                  example_signature=("signature_acc", lambda x: next((v for v in x if is_present(v)), "")),
                  )
             .reset_index()
             .sort_values(["unique_proteins", "hit_rows"], ascending=False))
    return out


def domains_per_protein(df_all: pd.DataFrame) -> pd.DataFrame:
    dpp = df_all.groupby("protein_id").size().reset_index(name="hit_rows")
    return dpp.sort_values("hit_rows", ascending=False)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Main InterProScan TSV")
    ap.add_argument("--dbcan", default=None, help="dbCAN InterProScan TSV (separate run)")
    ap.add_argument("--merops", default=None, help="MEROPS InterProScan TSV (separate run)")
    ap.add_argument("--fasta", default=None, help="Protein FASTA to compute total proteins and unannotated list")
    ap.add_argument("-o", "--outdir", default="results", help="Output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    safe_mkdir(outdir)

    # Read main
    df_main = read_tsv_flexible(Path(args.input), label="main")

    # Read optional dbCAN and MEROPS
    df_dbcan = read_tsv_flexible(Path(args.dbcan), label="dbCAN") if args.dbcan else None
    df_merops = read_tsv_flexible(Path(args.merops), label="MEROPS") if args.merops else None

    # Combine for global stats and keyword protease discovery
    frames = [df_main]
    if df_dbcan is not None:
        frames.append(df_dbcan)
    if df_merops is not None:
        frames.append(df_merops)
    df_all = pd.concat(frames, ignore_index=True)

    # FASTA IDs
    all_proteins = extract_ids_from_fasta(Path(args.fasta)) if args.fasta else None

    # Write outputs
    write_tsv(global_stats(df_all, all_proteins), outdir / "global_stats.tsv")
    write_tsv(analysis_contribution(df_all), outdir / "analysis_contribution.tsv")

    interpro_ab, sig_ab = abundance_tables(df_main)
    write_tsv(interpro_ab, outdir / "interpro_abundance.tsv")
    write_tsv(sig_ab, outdir / "signature_abundance.tsv")

    go_hit, go_prot = go_summaries(df_main)
    write_tsv(go_hit, outdir / "go_term_counts.tsv")
    write_tsv(go_prot, outdir / "go_protein_counts.tsv")

    c_hit, c_prot = cazymes(df_dbcan)
    write_tsv(c_hit, outdir / "cazymes_hit_counts.tsv")
    write_tsv(c_prot, outdir / "cazymes_protein_counts.tsv")

    p_hit, p_prot, p_class = proteases_keyword_table(df_all)
    write_tsv(p_hit, outdir / "protease_hit_counts.tsv")
    write_tsv(p_prot, outdir / "protease_protein_counts.tsv")
    write_tsv(p_class, outdir / "protease_class_counts.tsv")

    write_tsv(merops_families(df_merops), outdir / "protease_families.tsv")

    write_tsv(ec_numbers(df_main), outdir / "ec_numbers.tsv")
    write_tsv(domains_per_protein(df_all), outdir / "domains_per_protein.tsv")

    # Unannotated proteins list (needs FASTA)
    if all_proteins is not None:
        hit_proteins = pd.Index(df_all["protein_id"].unique())
        unannot = all_proteins.difference(hit_proteins)
        (outdir / "unannotated_proteins.ids").write_text("\n".join(unannot) + ("\n" if len(unannot) else ""))
        print("Wrote unannotated_proteins.ids:", len(unannot))

    print("Done. Wrote outputs to:", outdir.resolve())


if __name__ == "__main__":
    main()
