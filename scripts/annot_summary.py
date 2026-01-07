#!/usr/bin/env python3
"""
Summarize annotations from:
  1) InterProScan TSV (main) - 13/14/15 columns, no header
  2) dbCAN overview TSV (raw) - header: Gene ID, HMMER, Hotpep, DIAMOND, #ofTools
  3) MEROPS raw TSV (your format) - whitespace/tsv, no header

Outputs (TSV):
  global_stats.tsv
  analysis_contribution.tsv
  interpro_abundance.tsv
  signature_abundance.tsv
  go_term_counts.tsv
  go_protein_counts.tsv
  ec_numbers.tsv
  domains_per_protein.tsv
  unannotated_proteins.ids (if --fasta provided)

  cazymes_hit_counts.tsv
  cazymes_protein_counts.tsv
  protease_families.tsv
  protease_class_counts.tsv
  protease_hit_counts.tsv
  protease_protein_counts.tsv

Usage:
  python scripts/annot_summary.py \
    --interpro data/interproscan.tsv \
    --dbcan data/dbcan_overview.tsv \
    --merops data/merops.tsv \
    --fasta data/proteins.fasta \
    -o results
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

# -----------------------------
# Common helpers
# -----------------------------
MISSING = {"", "-", "NA", "NaN", "nan", "null", "None", "none"}

def norm_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s in MISSING else s

def is_present(x) -> bool:
    return norm_str(x) != ""

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_tsv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False)

def split_terms(s: str) -> List[str]:
    s = norm_str(s)
    if not s:
        return []
    parts = re.split(r"[|,;]+", s)
    return [p.strip() for p in parts if p.strip()]

def extract_ids_from_fasta(fasta: Path) -> pd.Index:
    ids = []
    with fasta.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                ids.append(line[1:].strip().split()[0])
    return pd.Index(pd.unique(ids))

# -----------------------------
# 1) InterProScan TSV reader
# -----------------------------
STD_COLS_15 = [
    "protein_id", "md5", "seq_len", "analysis",
    "signature_acc", "signature_desc",
    "start", "end", "score", "status", "date",
    "interpro_acc", "interpro_desc",
    "go_terms", "pathways"
]
STD_COLS_14 = STD_COLS_15[:-1]
STD_COLS_13 = STD_COLS_15[:-2]

def read_interproscan_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, na_filter=False)
    ncol = df.shape[1]

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
    else:
        raise ValueError(f"InterProScan TSV has {ncol} columns; expected 13–15.")

    # normalize key string cols
    for c in ["protein_id","analysis","signature_acc","signature_desc","interpro_acc","interpro_desc","go_terms","pathways"]:
        df[c] = df[c].map(norm_str)

    # numeric-ish
    df["seq_len"] = pd.to_numeric(df["seq_len"], errors="coerce")
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")

    df["source"] = "interproscan"
    return df

# -----------------------------
# 2) dbCAN overview reader (raw)
# -----------------------------
# Example:
# Gene ID   HMMER                         Hotpep     DIAMOND    #ofTools
# FUN_xxx   GT35(148-486)+GT35(491-923)   GT35(1)    ...        3
CAZY_FAM_RE = re.compile(r"\b(GH|GT|PL|CE|AA|CBM)(\d+)\b", re.I)

def extract_cazy_fams_from_cell(cell: str) -> List[str]:
    """
    Extract CAZy families from a single dbCAN cell like:
      'GT35(148-486)+GT35(491-923)'
      'GT2_Chitin_synth_2(108-287)+GT2_Chitin_synth_2(290-514)'  -> extracts GT2
      'GH37(184-820)' -> extracts GH37
    """
    cell = norm_str(cell)
    if not cell:
        return []
    hits = []
    for m in CAZY_FAM_RE.finditer(cell):
        hits.append(f"{m.group(1).upper()}{m.group(2)}")
    # unique but preserve order
    seen = set()
    out = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out

def read_dbcan_overview(path: Path) -> pd.DataFrame:
    # dbCAN overview usually has header
    df = pd.read_csv(path, sep="\t", dtype=str, na_filter=False)
    # Normalize columns for safety
    df.columns = [c.strip() for c in df.columns]

    # Accept slight header variations
    gene_col = None
    for cand in ["Gene ID", "GeneID", "Gene", "gene_id", "GeneID "]:
        if cand in df.columns:
            gene_col = cand
            break
    if gene_col is None:
        # try first column
        gene_col = df.columns[0]

    # ensure expected fields exist if present
    hmmer = df["HMMER"] if "HMMER" in df.columns else ""
    hotpep = df["Hotpep"] if "Hotpep" in df.columns else ""
    diamond = df["DIAMOND"] if "DIAMOND" in df.columns else ""

    rows = []
    for _, r in df.iterrows():
        pid = norm_str(r.get(gene_col, ""))
        if not pid:
            continue

        # Extract families from each tool column
        fams_h = extract_cazy_fams_from_cell(r.get("HMMER", ""))
        fams_p = extract_cazy_fams_from_cell(r.get("Hotpep", ""))
        fams_d = extract_cazy_fams_from_cell(r.get("DIAMOND", ""))

        # Record hits per tool (hit-level table)
        for fam in fams_h:
            rows.append((pid, "dbCAN_HMMER", fam, norm_str(r.get("HMMER",""))))
        for fam in fams_p:
            rows.append((pid, "dbCAN_Hotpep", fam, norm_str(r.get("Hotpep",""))))
        for fam in fams_d:
            rows.append((pid, "dbCAN_DIAMOND", fam, norm_str(r.get("DIAMOND",""))))

    out = pd.DataFrame(rows, columns=["protein_id","analysis","cazy_family","raw_hit"])
    return out

def cazymes_tables(df_dbcan_hits: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df_dbcan_hits is None or df_dbcan_hits.empty:
        hit_cols = ["cazy_family","analysis","hit_rows","unique_proteins"]
        prot_cols = ["cazy_family","unique_proteins"]
        return pd.DataFrame(columns=hit_cols), pd.DataFrame(columns=prot_cols)

    hit_counts = (df_dbcan_hits.groupby(["cazy_family","analysis"])
                  .agg(hit_rows=("cazy_family","size"),
                       unique_proteins=("protein_id","nunique"))
                  .reset_index()
                  .sort_values(["unique_proteins","hit_rows"], ascending=False))

    prot_counts = (df_dbcan_hits.groupby("cazy_family")["protein_id"]
                   .nunique()
                   .reset_index(name="unique_proteins")
                   .sort_values("unique_proteins", ascending=False))
    return hit_counts, prot_counts

# -----------------------------
# 3) MEROPS raw reader (your format)
# -----------------------------
# Example lines:
# FUN_000014-T1  MER0138795  Metallo  M  M24  355.1
# FUN_000038-T1  MER0417925  Metallo  M  M41  89.0
#
# Sometimes you may have 6 columns (no clan) or 7 columns (clan + family).
# We'll detect flexibly:
# col0 protein_id
# col1 merops_id
# col2 catalytic_type_name (Metallo/Serine/Cysteine/Aspartic/Threonine etc.)
# col3 catalytic_type_letter (M/S/C/A/T)
# remaining: family is last "M24/M41/S8/C1..." token before score if present
def read_merops_raw(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\t+|\s+", line)
            if len(parts) < 5:
                continue

            protein_id = norm_str(parts[0])
            merops_id = norm_str(parts[1])
            cat_name = norm_str(parts[2])
            cat_letter = norm_str(parts[3])

            # last token might be score (float). family often is token right before score.
            score = ""
            fam = ""
            clan = ""

            # try parse last as float
            last = parts[-1]
            if re.fullmatch(r"[0-9]+(\.[0-9]+)?", last):
                score = last
                # family candidate is previous token
                fam = norm_str(parts[-2]) if len(parts) >= 6 else ""
                # clan candidate if exists
                if len(parts) >= 7:
                    clan = norm_str(parts[-3])
            else:
                # no numeric score; treat last as family
                fam = norm_str(parts[-1])
                if len(parts) >= 6:
                    clan = norm_str(parts[-2])

            rows.append((protein_id, merops_id, cat_name, cat_letter, clan, fam, score))

    df = pd.DataFrame(rows, columns=[
        "protein_id","merops_id","catalytic_type","catalytic_letter","clan","family","score"
    ])
    # normalize family/catalytic type
    df["protein_id"] = df["protein_id"].map(norm_str)
    df["family"] = df["family"].map(norm_str)
    df["catalytic_type"] = df["catalytic_type"].map(norm_str)
    df["catalytic_letter"] = df["catalytic_letter"].map(norm_str)
    df["analysis"] = "MEROPS"
    return df[df["protein_id"].map(is_present)].copy()

def merops_family_table(df_merops: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df_merops is None or df_merops.empty:
        return pd.DataFrame(columns=["protease_family","hit_rows","unique_proteins","catalytic_type"])

    d = df_merops[df_merops["family"].map(is_present)].copy()
    if d.empty:
        return pd.DataFrame(columns=["protease_family","hit_rows","unique_proteins","catalytic_type"])

    tbl = (d.groupby(["family","catalytic_type"])
           .agg(hit_rows=("family","size"),
                unique_proteins=("protein_id","nunique"))
           .reset_index()
           .rename(columns={"family":"protease_family"})
           .sort_values(["unique_proteins","hit_rows"], ascending=False))
    return tbl

def merops_class_table(df_merops: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df_merops is None or df_merops.empty:
        return pd.DataFrame(columns=["protease_class","hit_rows","unique_proteins"])

    d = df_merops.copy()
    # prefer catalytic_type name, fallback to letter
    def klass(r):
        if is_present(r.get("catalytic_type","")):
            return r["catalytic_type"].lower()
        if is_present(r.get("catalytic_letter","")):
            return r["catalytic_letter"].upper()
        return "unknown"

    d["protease_class"] = d.apply(klass, axis=1)
    tbl = (d.groupby("protease_class")
           .agg(hit_rows=("protease_class","size"),
                unique_proteins=("protein_id","nunique"))
           .reset_index()
           .sort_values(["unique_proteins","hit_rows"], ascending=False))
    return tbl

# -----------------------------
# InterProScan summaries
# -----------------------------
def global_stats(df_all_hits: pd.DataFrame, all_proteins: Optional[pd.Index]) -> pd.DataFrame:
    proteins_with_any_hit = pd.Index(df_all_hits["protein_id"].unique())

    total_proteins = len(all_proteins) if all_proteins is not None else len(proteins_with_any_hit)
    proteins_with_hit = len(proteins_with_any_hit)
    proteins_without_hit = (total_proteins - proteins_with_hit) if all_proteins is not None else np.nan

    # For these, use only interproscan rows if columns exist
    if "interpro_acc" in df_all_hits.columns:
        proteins_with_interpro = df_all_hits.loc[df_all_hits.get("interpro_acc","").map(is_present), "protein_id"].nunique()
        proteins_with_go = df_all_hits.loc[df_all_hits.get("go_terms","").map(is_present), "protein_id"].nunique()
        proteins_with_pathway = df_all_hits.loc[df_all_hits.get("pathways","").map(is_present), "protein_id"].nunique()
        hit_rows_with_interpro = int(df_all_hits.get("interpro_acc","").map(is_present).sum())
        hit_rows_with_go = int(df_all_hits.get("go_terms","").map(is_present).sum())
        hit_rows_with_pathway = int(df_all_hits.get("pathways","").map(is_present).sum())
    else:
        proteins_with_interpro = proteins_with_go = proteins_with_pathway = 0
        hit_rows_with_interpro = hit_rows_with_go = hit_rows_with_pathway = 0

    dpp = df_all_hits.groupby("protein_id").size()

    out = pd.DataFrame([{
        "total_proteins": total_proteins,
        "proteins_with_any_hit": proteins_with_hit,
        "proteins_without_hit": proteins_without_hit,
        "total_hit_rows": len(df_all_hits),
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

def analysis_contribution(df_interpro: pd.DataFrame,
                          df_dbcan_hits: Optional[pd.DataFrame],
                          df_merops: Optional[pd.DataFrame]) -> pd.DataFrame:
    rows = []

    # InterProScan contribution
    g = df_interpro.groupby("analysis")
    tmp = pd.DataFrame({
        "analysis": g.size().index,
        "hit_rows": g.size().values,
        "unique_proteins": g["protein_id"].nunique().values,
        "unique_signatures": g["signature_acc"].apply(lambda x: len({v for v in x if is_present(v)})).values,
    })
    rows.append(tmp)

    # dbCAN contribution
    if df_dbcan_hits is not None and not df_dbcan_hits.empty:
        g2 = df_dbcan_hits.groupby("analysis")
        tmp2 = pd.DataFrame({
            "analysis": g2.size().index,
            "hit_rows": g2.size().values,
            "unique_proteins": g2["protein_id"].nunique().values,
            "unique_signatures": g2["cazy_family"].nunique().values,  # CAZy families
        })
        rows.append(tmp2)

    # MEROPS contribution
    if df_merops is not None and not df_merops.empty:
        g3 = df_merops.groupby("analysis")
        tmp3 = pd.DataFrame({
            "analysis": g3.size().index,
            "hit_rows": g3.size().values,
            "unique_proteins": g3["protein_id"].nunique().values,
            "unique_signatures": g3["family"].nunique().values,
        })
        rows.append(tmp3)

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values("hit_rows", ascending=False)
    return out

def abundance_tables(df_interpro: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # InterPro abundance (exclude missing)
    ip = df_interpro[df_interpro["interpro_acc"].map(is_present)].copy()
    interpro_ab = (
        ip.groupby(["interpro_acc","interpro_desc"])
          .agg(hit_rows=("interpro_acc","size"),
               unique_proteins=("protein_id","nunique"))
          .reset_index()
          .sort_values(["unique_proteins","hit_rows"], ascending=False)
    )

    # Signature abundance
    sig = df_interpro[df_interpro["signature_acc"].map(is_present)].copy()
    sig_ab = (
        sig.groupby(["analysis","signature_acc","signature_desc"])
           .agg(hit_rows=("signature_acc","size"),
                unique_proteins=("protein_id","nunique"))
           .reset_index()
           .sort_values(["unique_proteins","hit_rows"], ascending=False)
    )
    return interpro_ab, sig_ab

def go_summaries(df_interpro: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, r in df_interpro.iterrows():
        for go in split_terms(r.get("go_terms","")):
            if go.startswith("GO:"):
                rows.append((go, r["protein_id"]))
    if not rows:
        return (pd.DataFrame(columns=["go_term","hit_rows","unique_proteins"]),
                pd.DataFrame(columns=["go_term","unique_proteins"]))
    go_df = pd.DataFrame(rows, columns=["go_term","protein_id"])
    go_hit = (go_df.groupby("go_term")
              .agg(hit_rows=("go_term","size"),
                   unique_proteins=("protein_id","nunique"))
              .reset_index()
              .sort_values(["unique_proteins","hit_rows"], ascending=False))
    go_prot = (go_df.groupby("go_term")["protein_id"]
               .nunique()
               .reset_index(name="unique_proteins")
               .sort_values("unique_proteins", ascending=False))
    return go_hit, go_prot

EC_RE = re.compile(r"\bEC[:\s]*([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)\b")

def ec_numbers(df_interpro: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_interpro.iterrows():
        blob = " ".join([
            str(r.get("signature_desc","")),
            str(r.get("interpro_desc","")),
            str(r.get("pathways","")),
        ])
        ecs = EC_RE.findall(blob)
        if not ecs:
            continue
        for ec in set(ecs):
            rows.append((ec, r["protein_id"], r.get("analysis",""),
                         r.get("interpro_acc",""), r.get("signature_acc","")))
    if not rows:
        return pd.DataFrame(columns=["ec_number","unique_proteins","hit_rows","analyses","example_interpro","example_signature"])

    e = pd.DataFrame(rows, columns=["ec_number","protein_id","analysis","interpro_acc","signature_acc"])
    out = (e.groupby("ec_number")
           .agg(unique_proteins=("protein_id","nunique"),
                hit_rows=("ec_number","size"),
                analyses=("analysis", lambda x: ",".join(sorted(set([a for a in x if a])))),
                example_interpro=("interpro_acc", lambda x: next((v for v in x if is_present(v)), "")),
                example_signature=("signature_acc", lambda x: next((v for v in x if is_present(v)), "")))
           .reset_index()
           .sort_values(["unique_proteins","hit_rows"], ascending=False))
    return out

def domains_per_protein(df_all_hits: pd.DataFrame) -> pd.DataFrame:
    dpp = df_all_hits.groupby("protein_id").size().reset_index(name="hit_rows")
    return dpp.sort_values("hit_rows", ascending=False)

# Keyword protease tables from InterProScan (optional but helpful)
PROTEASE_CLASS_PATTERNS = [
    ("serine_protease", re.compile(r"\bserine\b|\bsubtilisin\b|\btrypsin\b", re.I)),
    ("metalloprotease", re.compile(r"\bmetallo\b|\bzinc\b|\bmatrix metalloproteinase\b", re.I)),
    ("cysteine_protease", re.compile(r"\bcysteine\b|\bpapain\b|\bcathepsin\b", re.I)),
    ("aspartic_protease", re.compile(r"\baspartic\b|\bpepsin\b", re.I)),
    ("threonine_protease", re.compile(r"\bthreonine\b|\bproteasome\b", re.I)),
]

def interproscan_protease_tables(df_interpro: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    blob = (df_interpro["signature_desc"].fillna("") + " " + df_interpro["interpro_desc"].fillna("")).astype(str)
    is_prot = blob.str.contains(r"\b(peptidase|protease)\b", case=False, regex=True) | \
              df_interpro["interpro_desc"].astype(str).str.contains(r"\bpeptidase\b", case=False, regex=True)

    p = df_interpro[is_prot].copy()
    if p.empty:
        base_cols = ["protease_key","analysis","signature_desc","interpro_acc","interpro_desc","hit_rows","unique_proteins"]
        return pd.DataFrame(columns=base_cols), pd.DataFrame(columns=["protease_key","unique_proteins"])

    p["protease_key"] = np.where(p["interpro_acc"].map(is_present), p["interpro_acc"], p["signature_acc"])

    hit_counts = (p.groupby(["protease_key","analysis","signature_desc","interpro_acc","interpro_desc"])
                  .agg(hit_rows=("protease_key","size"),
                       unique_proteins=("protein_id","nunique"))
                  .reset_index()
                  .sort_values(["unique_proteins","hit_rows"], ascending=False))

    prot_counts = (p.groupby("protease_key")["protein_id"]
                   .nunique()
                   .reset_index(name="unique_proteins")
                   .sort_values("unique_proteins", ascending=False))
    return hit_counts, prot_counts

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interpro", required=True, help="Main InterProScan TSV (13–15 cols, no header)")
    ap.add_argument("--dbcan", default=None, help="dbCAN overview TSV (raw, with Gene ID/HMMER/Hotpep/DIAMOND/#ofTools)")
    ap.add_argument("--merops", default=None, help="MEROPS raw file (your format)")
    ap.add_argument("--fasta", default=None, help="Proteins FASTA for total proteins and unannotated list")
    ap.add_argument("-o", "--outdir", default="results", help="Output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    safe_mkdir(outdir)

    # Read InterProScan
    df_interpro = read_interproscan_tsv(Path(args.interpro))

    # Read dbCAN (raw overview) -> hit table
    df_dbcan_hits = read_dbcan_overview(Path(args.dbcan)) if args.dbcan else None

    # Read MEROPS raw
    df_merops = read_merops_raw(Path(args.merops)) if args.merops else None

    # Build "all hits" table for global/domains-per-protein:
    # - InterProScan already has protein_id rows
    # - dbCAN hits: one row per protein-family-tool
    # - MEROPS hits: one row per protein (or per protein-family)
    frames = [df_interpro[["protein_id","analysis","interpro_acc","go_terms","pathways"]].copy()]
    frames[0]["hit_source"] = "interproscan"

    if df_dbcan_hits is not None and not df_dbcan_hits.empty:
        t = df_dbcan_hits[["protein_id","analysis"]].copy()
        t["interpro_acc"] = ""
        t["go_terms"] = ""
        t["pathways"] = ""
        t["hit_source"] = "dbcan"
        frames.append(t)

    if df_merops is not None and not df_merops.empty:
        t = df_merops[["protein_id","analysis"]].copy()
        t["interpro_acc"] = ""
        t["go_terms"] = ""
        t["pathways"] = ""
        t["hit_source"] = "merops"
        frames.append(t)

    df_all_hits = pd.concat(frames, ignore_index=True)

    # FASTA IDs
    all_proteins = extract_ids_from_fasta(Path(args.fasta)) if args.fasta else None

    # Outputs
    write_tsv(global_stats(df_all_hits, all_proteins), outdir / "global_stats.tsv")
    write_tsv(analysis_contribution(df_interpro, df_dbcan_hits, df_merops), outdir / "analysis_contribution.tsv")

    interpro_ab, sig_ab = abundance_tables(df_interpro)
    write_tsv(interpro_ab, outdir / "interpro_abundance.tsv")
    write_tsv(sig_ab, outdir / "signature_abundance.tsv")

    go_hit, go_prot = go_summaries(df_interpro)
    write_tsv(go_hit, outdir / "go_term_counts.tsv")
    write_tsv(go_prot, outdir / "go_protein_counts.tsv")

    write_tsv(ec_numbers(df_interpro), outdir / "ec_numbers.tsv")
    write_tsv(domains_per_protein(df_all_hits), outdir / "domains_per_protein.tsv")

    # CAZymes
    c_hit, c_prot = cazymes_tables(df_dbcan_hits)
    write_tsv(c_hit, outdir / "cazymes_hit_counts.tsv")
    write_tsv(c_prot, outdir / "cazymes_protein_counts.tsv")

    # Proteases: MEROPS families + class counts
    write_tsv(merops_family_table(df_merops), outdir / "protease_families.tsv")
    write_tsv(merops_class_table(df_merops), outdir / "protease_class_counts.tsv")

    # Proteases from InterProScan keywords (additional)
    p_hit, p_prot = interproscan_protease_tables(df_interpro)
    write_tsv(p_hit, outdir / "protease_hit_counts.tsv")
    write_tsv(p_prot, outdir / "protease_protein_counts.tsv")

    # Unannotated proteins list
    if all_proteins is not None:
        hit_proteins = pd.Index(df_all_hits["protein_id"].unique())
        unannot = all_proteins.difference(hit_proteins)
        (outdir / "unannotated_proteins.ids").write_text("\n".join(unannot) + ("\n" if len(unannot) else ""))
        print("Wrote unannotated_proteins.ids:", len(unannot))

    print("Done. Wrote outputs to:", outdir.resolve())

if __name__ == "__main__":
    main()
